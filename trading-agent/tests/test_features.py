"""Tests de la ingeniería de características y validación de datos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_agent.data.provider import validate_ohlcv
from trading_agent.exceptions import DataValidationError
from trading_agent.features import FEATURE_COLUMNS, build_features


class TestValidateOhlcv:
    def test_valid_frame_passes(self, ohlcv):
        out = validate_ohlcv(ohlcv, "TEST")
        assert list(out.columns) == list(("open", "high", "low", "close", "volume"))
        assert str(out.index.tz) == "UTC"

    def test_empty_frame_raises(self):
        with pytest.raises(DataValidationError, match="vacíos"):
            validate_ohlcv(pd.DataFrame(), "TEST")

    def test_missing_column_raises(self, ohlcv):
        with pytest.raises(DataValidationError, match="faltan columnas"):
            validate_ohlcv(ohlcv.drop(columns=["close"]), "TEST")

    def test_negative_price_raises(self, ohlcv):
        bad = ohlcv.copy()
        bad.iloc[5, bad.columns.get_loc("close")] = -1.0
        with pytest.raises(DataValidationError, match="precios <= 0"):
            validate_ohlcv(bad, "TEST")

    def test_nan_rows_are_dropped(self, ohlcv):
        dirty = ohlcv.copy()
        dirty.iloc[3, dirty.columns.get_loc("close")] = np.nan
        out = validate_ohlcv(dirty, "TEST")
        assert len(out) == len(ohlcv) - 1

    def test_duplicated_index_keeps_last(self, ohlcv):
        dup = pd.concat([ohlcv, ohlcv.iloc[[0]]])
        out = validate_ohlcv(dup, "TEST")
        assert not out.index.duplicated().any()
        assert out.index.is_monotonic_increasing

    def test_naive_index_gets_utc(self, ohlcv):
        naive = ohlcv.copy()
        naive.index = naive.index.tz_localize(None)
        out = validate_ohlcv(naive, "TEST")
        assert str(out.index.tz) == "UTC"


class TestBuildFeatures:
    def test_columns_and_no_nan(self, ohlcv):
        feats = build_features(ohlcv)
        assert tuple(feats.columns) == FEATURE_COLUMNS
        assert not feats.isna().any().any()

    def test_features_are_bounded(self, ohlcv):
        """Los indicadores acotados deben respetar sus rangos de diseño."""
        feats = build_features(ohlcv)
        assert feats["rsi"].between(-1.0, 1.0).all()
        assert feats["bb_pos"].between(-3.0, 3.0).all()
        assert feats["volume_z"].between(-3.0, 3.0).all()

    def test_too_short_history_raises(self, ohlcv):
        with pytest.raises(DataValidationError, match="Se necesitan"):
            build_features(ohlcv.head(20))

    def test_bad_macd_params_raise(self, ohlcv):
        with pytest.raises(DataValidationError, match="macd_slow"):
            build_features(ohlcv, macd_fast=26, macd_slow=12)

    def test_log_return_matches_manual_computation(self, ohlcv):
        feats = build_features(ohlcv)
        ts = feats.index[5]
        i = ohlcv.index.get_loc(ts)
        expected = np.log(ohlcv["close"].iloc[i] / ohlcv["close"].iloc[i - 1])
        assert feats.loc[ts, "log_ret"] == pytest.approx(expected)

    def test_deterministic(self, ohlcv):
        pd.testing.assert_frame_equal(build_features(ohlcv),
                                      build_features(ohlcv))
