"""
tests/trm_v2/test_config_loader_trm_v2.py

Tests for the TRM v2 config_loader extension:
  - load_trm_v2_config() returns correct typed dataclasses
  - All flat accessors return their defaults when config tables are absent
  - Environment variable overrides work for each section
  - _Cfg singleton exposes all new methods
"""

from __future__ import annotations

import os
import tempfile
import textwrap
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loader(toml_content: str):
    """Return a fresh config_loader module whose _CONFIG_PATH points to a temp TOML file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".toml", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(toml_content)
        tmp_path = Path(fh.name)

    import importlib
    import mycelium.pipeline.config_loader as _orig
    with mock.patch.object(type(_orig), "_CONFIG_PATH", new=tmp_path, create=True):
        # Re-import with patched path by monkeypatching at module level
        import mycelium.pipeline.config_loader as loader
        old_path = loader._CONFIG_PATH
        loader._CONFIG_PATH = tmp_path
        yield loader
        loader._CONFIG_PATH = old_path

    tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# load_trm_v2_config() — defaults when tables absent
# ---------------------------------------------------------------------------

class TestLoadTRMV2ConfigDefaults:
    def test_returns_trmv2fullconfig(self):
        from mycelium.pipeline.config_loader import load_trm_v2_config, TRMv2FullConfig
        result = load_trm_v2_config()
        assert isinstance(result, TRMv2FullConfig)

    def test_trm_v2_defaults(self):
        from mycelium.pipeline.config_loader import load_trm_v2_config
        cfg = load_trm_v2_config()
        assert cfg.trm_v2.enabled is False
        assert cfg.trm_v2.gate_top_k == 3
        assert cfg.trm_v2.gate_mode == "frozen"
        assert "trm_v2" in cfg.trm_v2.encoder_checkpoint
        assert "trm_v2" in cfg.trm_v2.gate_checkpoint

    def test_domain_graph_defaults(self):
        from mycelium.pipeline.config_loader import load_trm_v2_config
        cfg = load_trm_v2_config()
        assert cfg.domain_graph.enabled is False
        assert "domain_graph" in cfg.domain_graph.graph_path
        assert cfg.domain_graph.graph_schema_version == "1.0"

    def test_sqlite_experts_defaults(self):
        from mycelium.pipeline.config_loader import load_trm_v2_config
        cfg = load_trm_v2_config()
        assert cfg.sqlite_experts.enabled is False
        assert "sqlite_experts" in cfg.sqlite_experts.root_dir
        assert cfg.sqlite_experts.use_fts5 is True
        assert cfg.sqlite_experts.vector_extension == "auto"

    def test_cold_storage_defaults(self):
        from mycelium.pipeline.config_loader import load_trm_v2_config
        cfg = load_trm_v2_config()
        assert cfg.cold_storage.enabled is False
        assert cfg.cold_storage.quantize_on_store is True
        assert cfg.cold_storage.replay_min_samples == 500
        assert cfg.cold_storage.replay_max_samples == 1000
        assert cfg.cold_storage.reactivation_queue is True


# ---------------------------------------------------------------------------
# Flat accessors — defaults
# ---------------------------------------------------------------------------

class TestFlatAccessorDefaults:
    def test_trm_v2_enabled_default_false(self):
        from mycelium.pipeline.config_loader import trm_v2_enabled
        assert trm_v2_enabled() is False

    def test_trm_v2_gate_top_k_default(self):
        from mycelium.pipeline.config_loader import trm_v2_gate_top_k
        assert trm_v2_gate_top_k() == 3

    def test_trm_v2_gate_mode_default(self):
        from mycelium.pipeline.config_loader import trm_v2_gate_mode
        assert trm_v2_gate_mode() == "frozen"

    def test_domain_graph_enabled_default_false(self):
        from mycelium.pipeline.config_loader import domain_graph_enabled
        assert domain_graph_enabled() is False

    def test_domain_graph_schema_version_default(self):
        from mycelium.pipeline.config_loader import domain_graph_schema_version
        assert domain_graph_schema_version() == "1.0"

    def test_sqlite_experts_enabled_default_false(self):
        from mycelium.pipeline.config_loader import sqlite_experts_enabled
        assert sqlite_experts_enabled() is False

    def test_sqlite_experts_use_fts5_default_true(self):
        from mycelium.pipeline.config_loader import sqlite_experts_use_fts5
        assert sqlite_experts_use_fts5() is True

    def test_cold_storage_enabled_default_false(self):
        from mycelium.pipeline.config_loader import cold_storage_enabled
        assert cold_storage_enabled() is False

    def test_cold_storage_replay_min_samples_default(self):
        from mycelium.pipeline.config_loader import cold_storage_replay_min_samples
        assert cold_storage_replay_min_samples() == 500

    def test_cold_storage_replay_max_samples_default(self):
        from mycelium.pipeline.config_loader import cold_storage_replay_max_samples
        assert cold_storage_replay_max_samples() == 1000


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------

class TestEnvVarOverrides:
    def test_trm_v2_enabled_env(self):
        from mycelium.pipeline.config_loader import trm_v2_enabled
        with mock.patch.dict(os.environ, {"MYCELIUM_TRM_V2_ENABLED": "true"}):
            assert trm_v2_enabled() is True
        assert trm_v2_enabled() is False  # restored

    def test_trm_v2_gate_top_k_env(self):
        from mycelium.pipeline.config_loader import trm_v2_gate_top_k
        with mock.patch.dict(os.environ, {"MYCELIUM_TRM_V2_GATE_TOP_K": "7"}):
            assert trm_v2_gate_top_k() == 7

    def test_domain_graph_enabled_env(self):
        from mycelium.pipeline.config_loader import domain_graph_enabled
        with mock.patch.dict(os.environ, {"MYCELIUM_DOMAIN_GRAPH_ENABLED": "1"}):
            assert domain_graph_enabled() is True

    def test_sqlite_experts_enabled_env(self):
        from mycelium.pipeline.config_loader import sqlite_experts_enabled
        with mock.patch.dict(os.environ, {"MYCELIUM_SQLITE_EXPERTS_ENABLED": "yes"}):
            assert sqlite_experts_enabled() is True

    def test_cold_storage_enabled_env(self):
        from mycelium.pipeline.config_loader import cold_storage_enabled
        with mock.patch.dict(os.environ, {"MYCELIUM_COLD_STORAGE_ENABLED": "true"}):
            assert cold_storage_enabled() is True

    def test_cold_storage_replay_min_samples_env(self):
        from mycelium.pipeline.config_loader import cold_storage_replay_min_samples
        with mock.patch.dict(os.environ, {"MYCELIUM_COLD_STORAGE_REPLAY_MIN_SAMPLES": "250"}):
            assert cold_storage_replay_min_samples() == 250


# ---------------------------------------------------------------------------
# _Cfg singleton exposes all new methods
# ---------------------------------------------------------------------------

class TestCfgSingleton:
    def test_singleton_has_load_trm_v2_config(self):
        from mycelium.pipeline.config_loader import cfg
        assert callable(cfg.load_trm_v2_config)

    def test_singleton_has_trm_v2_enabled(self):
        from mycelium.pipeline.config_loader import cfg
        assert callable(cfg.trm_v2_enabled)

    def test_singleton_has_domain_graph_enabled(self):
        from mycelium.pipeline.config_loader import cfg
        assert callable(cfg.domain_graph_enabled)

    def test_singleton_has_sqlite_experts_enabled(self):
        from mycelium.pipeline.config_loader import cfg
        assert callable(cfg.sqlite_experts_enabled)

    def test_singleton_has_cold_storage_enabled(self):
        from mycelium.pipeline.config_loader import cfg
        assert callable(cfg.cold_storage_enabled)

    def test_singleton_load_returns_full_config(self):
        from mycelium.pipeline.config_loader import cfg, TRMv2FullConfig
        result = cfg.load_trm_v2_config()
        assert isinstance(result, TRMv2FullConfig)

    def test_all_cold_storage_accessors_on_singleton(self):
        from mycelium.pipeline.config_loader import cfg
        assert callable(cfg.cold_storage_quantize_on_store)
        assert callable(cfg.cold_storage_replay_min_samples)
        assert callable(cfg.cold_storage_replay_max_samples)
        assert callable(cfg.cold_storage_reactivation_queue)

    def test_all_sqlite_experts_accessors_on_singleton(self):
        from mycelium.pipeline.config_loader import cfg
        assert callable(cfg.sqlite_experts_root_dir)
        assert callable(cfg.sqlite_experts_use_fts5)
        assert callable(cfg.sqlite_experts_vector_extension)

    def test_all_trm_v2_accessors_on_singleton(self):
        from mycelium.pipeline.config_loader import cfg
        assert callable(cfg.trm_v2_encoder_checkpoint)
        assert callable(cfg.trm_v2_gate_checkpoint)
        assert callable(cfg.trm_v2_gate_top_k)
        assert callable(cfg.trm_v2_gate_mode)

    def test_all_domain_graph_accessors_on_singleton(self):
        from mycelium.pipeline.config_loader import cfg
        assert callable(cfg.domain_graph_path)
        assert callable(cfg.domain_graph_schema_version)


# ---------------------------------------------------------------------------
# Dataclass field types
# ---------------------------------------------------------------------------

class TestDataclassFieldTypes:
    def test_trmv2config_fields(self):
        from mycelium.pipeline.config_loader import TRMV2Config
        c = TRMV2Config()
        assert isinstance(c.enabled, bool)
        assert isinstance(c.gate_top_k, int)
        assert isinstance(c.gate_mode, str)
        assert isinstance(c.encoder_checkpoint, str)
        assert isinstance(c.gate_checkpoint, str)

    def test_domain_graph_config_fields(self):
        from mycelium.pipeline.config_loader import DomainGraphConfig
        c = DomainGraphConfig()
        assert isinstance(c.enabled, bool)
        assert isinstance(c.graph_path, str)
        assert isinstance(c.graph_schema_version, str)

    def test_sqlite_experts_config_fields(self):
        from mycelium.pipeline.config_loader import SQLiteExpertsConfig
        c = SQLiteExpertsConfig()
        assert isinstance(c.enabled, bool)
        assert isinstance(c.root_dir, str)
        assert isinstance(c.use_fts5, bool)
        assert isinstance(c.vector_extension, str)

    def test_cold_storage_config_fields(self):
        from mycelium.pipeline.config_loader import ColdStorageConfig
        c = ColdStorageConfig()
        assert isinstance(c.enabled, bool)
        assert isinstance(c.quantize_on_store, bool)
        assert isinstance(c.replay_min_samples, int)
        assert isinstance(c.replay_max_samples, int)
        assert isinstance(c.reactivation_queue, bool)

    def test_full_config_sub_configs(self):
        from mycelium.pipeline.config_loader import (
            TRMv2FullConfig, TRMV2Config, DomainGraphConfig,
            SQLiteExpertsConfig, ColdStorageConfig,
        )
        c = TRMv2FullConfig()
        assert isinstance(c.trm_v2, TRMV2Config)
        assert isinstance(c.domain_graph, DomainGraphConfig)
        assert isinstance(c.sqlite_experts, SQLiteExpertsConfig)
        assert isinstance(c.cold_storage, ColdStorageConfig)
