"""Tests de los CLIs y utilidades de operación (sin red, sin torch)."""

from __future__ import annotations

import datetime as dt
import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


class TestHealthcheck:
    def _load(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "healthcheck", SCRIPTS / "healthcheck.py")
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod

    def _write(self, path: Path, **overrides) -> None:
        payload = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "iteration": 7, "consecutive_errors": 0, "halted": False}
        payload.update(overrides)
        path.write_text(json.dumps(payload))

    def test_fresh_heartbeat_is_healthy(self, tmp_path):
        hc = self._load()
        path = tmp_path / "heartbeat.json"
        self._write(path)
        ok, why = hc.check(path, max_age_s=300, max_errors=3)
        assert ok and "ok" in why

    def test_stale_heartbeat_is_unhealthy(self, tmp_path):
        hc = self._load()
        path = tmp_path / "heartbeat.json"
        old = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=30)
        self._write(path, utc=old.isoformat())
        ok, why = hc.check(path, max_age_s=300, max_errors=3)
        assert not ok and "hace" in why

    def test_missing_or_corrupt_heartbeat(self, tmp_path):
        hc = self._load()
        assert hc.check(tmp_path / "nope.json", max_age_s=1, max_errors=1)[0] is False
        bad = tmp_path / "bad.json"
        bad.write_text("{no json")
        assert hc.check(bad, max_age_s=1, max_errors=1)[0] is False

    def test_kill_switch_and_errors_are_unhealthy(self, tmp_path):
        hc = self._load()
        path = tmp_path / "heartbeat.json"
        self._write(path, halted=True)
        assert hc.check(path, max_age_s=300, max_errors=3)[0] is False
        self._write(path, consecutive_errors=5)
        assert hc.check(path, max_age_s=300, max_errors=3)[0] is False

    def test_cli_exit_codes(self, tmp_path):
        path = tmp_path / "heartbeat.json"
        self._write(path)
        ok = subprocess.run([sys.executable, str(SCRIPTS / "healthcheck.py"),
                             "--heartbeat", str(path)],
                            check=False, capture_output=True)
        assert ok.returncode == 0
        bad = subprocess.run([sys.executable, str(SCRIPTS / "healthcheck.py"),
                              "--heartbeat", str(tmp_path / "missing.json")],
                             check=False, capture_output=True)
        assert bad.returncode == 1


class TestLoggingSetup:
    def test_console_and_rotating_file(self, tmp_path):
        from trading_agent.logging_setup import configure_logging

        log_file = tmp_path / "logs" / "trader.log"
        configure_logging("INFO", log_file)
        logging.getLogger("x").info("mensaje de prueba")
        assert "mensaje de prueba" in log_file.read_text()
        kinds = {type(h).__name__ for h in logging.getLogger().handlers}
        assert "TimedRotatingFileHandler" in kinds and "StreamHandler" in kinds
        # Limpieza: deja el logging raíz como lo encontró pytest.
        configure_logging("WARNING", None)

    def test_invalid_level_raises(self):
        from trading_agent.logging_setup import configure_logging

        with pytest.raises(ValueError):
            configure_logging("RUIDOSO", None)
