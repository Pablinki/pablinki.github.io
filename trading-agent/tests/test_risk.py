"""Tests del gestor de riesgo: la última línea de defensa del capital."""

from __future__ import annotations

import pytest

from trading_agent.config import RiskConfig
from trading_agent.portfolio import Position
from trading_agent.risk import RiskManager


def make_rm(**kwargs) -> RiskManager:
    return RiskManager(RiskConfig(**kwargs))


class TestPositionSize:
    def test_respects_max_position_pct(self):
        rm = make_rm(max_position_pct=0.25)
        # 25% de 100k = 25k; a 100 USD/acción caben 250 acciones.
        assert rm.position_size(equity=100_000.0, price=100.0) == 250.0

    def test_rounds_down_to_whole_shares(self):
        rm = make_rm(max_position_pct=0.25)
        assert rm.position_size(equity=1_000.0, price=99.0) == 2.0  # 250/99=2.52

    @pytest.mark.parametrize("equity,price", [(0.0, 100.0), (100.0, 0.0),
                                              (-5.0, 100.0)])
    def test_degenerate_inputs_give_zero(self, equity, price):
        assert make_rm().position_size(equity, price) == 0.0


class TestForcedClose:
    def test_stop_loss_triggers(self):
        rm = make_rm(stop_loss_pct=0.05)
        pos = Position("AAPL", 10, entry_price=100.0)
        assert rm.should_force_close(pos, price=94.9) == "stop_loss"

    def test_take_profit_triggers(self):
        rm = make_rm(take_profit_pct=0.10)
        pos = Position("AAPL", 10, entry_price=100.0)
        assert rm.should_force_close(pos, price=110.1) == "take_profit"

    def test_take_profit_disabled_with_none(self):
        rm = make_rm(take_profit_pct=None)
        pos = Position("AAPL", 10, entry_price=100.0)
        assert rm.should_force_close(pos, price=500.0) is None

    def test_no_trigger_in_normal_range(self):
        rm = make_rm(stop_loss_pct=0.05, take_profit_pct=0.10)
        pos = Position("AAPL", 10, entry_price=100.0)
        assert rm.should_force_close(pos, price=102.0) is None


class TestKillSwitch:
    def test_drawdown_halts_trading(self):
        rm = make_rm(max_drawdown_pct=0.20)
        rm.update_equity(100_000.0)   # pico
        rm.update_equity(79_000.0)    # -21% > 20%
        assert rm.halted
        assert not rm.can_open(79_000.0)

    def test_within_drawdown_keeps_trading(self):
        rm = make_rm(max_drawdown_pct=0.20)
        rm.update_equity(100_000.0)
        rm.update_equity(85_000.0)    # -15%
        assert not rm.halted
        assert rm.can_open(85_000.0)

    def test_reset_rearms_after_halt(self):
        rm = make_rm(max_drawdown_pct=0.20)
        rm.update_equity(100_000.0)
        rm.update_equity(50_000.0)
        assert rm.halted
        rm.reset()
        assert not rm.halted


class TestDailyLossLimit:
    def test_daily_loss_blocks_new_positions(self):
        rm = make_rm(max_daily_loss_pct=0.03)
        rm.start_new_day(100_000.0)
        assert not rm.can_open(96_000.0)   # -4% en el día

    def test_small_daily_loss_allows_opening(self):
        rm = make_rm(max_daily_loss_pct=0.03)
        rm.start_new_day(100_000.0)
        assert rm.can_open(98_000.0)       # -2%

    def test_new_day_resets_the_limit(self):
        rm = make_rm(max_daily_loss_pct=0.03)
        rm.start_new_day(100_000.0)
        assert not rm.can_open(96_000.0)
        rm.start_new_day(96_000.0)          # jornada nueva
        assert rm.can_open(96_000.0)
