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


class TestIntrabarEvaluation:
    """Los stops se evalúan contra el rango completo de la vela."""

    def test_low_crossing_stop_fills_at_stop_level(self):
        rm = make_rm(stop_loss_pct=0.05, take_profit_pct=0.10)
        pos = Position("AAPL", 10, entry_price=100.0)
        reason, fill = rm.evaluate_bar(pos, open_=99.0, high=101.0, low=94.0,
                                       close=98.0)
        assert reason == "stop_loss"
        assert fill == pytest.approx(95.0)

    def test_gap_down_fills_at_open(self):
        rm = make_rm(stop_loss_pct=0.05, take_profit_pct=None)
        pos = Position("AAPL", 10, entry_price=100.0)
        reason, fill = rm.evaluate_bar(pos, open_=90.0, high=92.0, low=89.0,
                                       close=91.0)
        assert reason == "stop_loss" and fill == pytest.approx(90.0)

    def test_high_crossing_take_profit(self):
        rm = make_rm(stop_loss_pct=0.05, take_profit_pct=0.10)
        pos = Position("AAPL", 10, entry_price=100.0)
        reason, fill = rm.evaluate_bar(pos, open_=101.0, high=112.0, low=100.5,
                                       close=108.0)
        assert reason == "take_profit" and fill == pytest.approx(110.0)

    def test_gap_up_through_take_profit_fills_at_open_before_intrabar_stop(self):
        """Si la vela ya abre por encima del TP, se ejecutó al abrir: un
        mínimo posterior bajo el stop ya no afecta a esa posición."""
        rm = make_rm(stop_loss_pct=0.05, take_profit_pct=0.10)
        pos = Position("AAPL", 10, entry_price=100.0)
        reason, fill = rm.evaluate_bar(pos, open_=112.0, high=113.0, low=90.0,
                                       close=95.0)
        assert reason == "take_profit" and fill == pytest.approx(112.0)

    def test_stop_prevails_when_both_hit(self):
        rm = make_rm(stop_loss_pct=0.05, take_profit_pct=0.10)
        pos = Position("AAPL", 10, entry_price=100.0)
        reason, _ = rm.evaluate_bar(pos, open_=100.0, high=115.0, low=90.0,
                                    close=100.0)
        assert reason == "stop_loss"

    def test_no_trigger_returns_close(self):
        rm = make_rm(stop_loss_pct=0.05, take_profit_pct=0.10)
        pos = Position("AAPL", 10, entry_price=100.0)
        assert rm.evaluate_bar(pos, 100.0, 103.0, 98.0, 102.0) == (None, 102.0)


class TestCooldownAndPersistence:
    def test_cooldown_blocks_reentry_for_n_bars(self):
        rm = make_rm(cooldown_bars=2)
        rm.on_forced_close()
        assert not rm.can_open(100_000.0)
        rm.new_bar()
        assert not rm.can_open(100_000.0)
        rm.new_bar()
        assert rm.can_open(100_000.0)

    def test_cooldown_disabled_by_default(self):
        rm = make_rm()
        rm.on_forced_close()
        assert rm.can_open(100_000.0)

    def test_state_roundtrip_keeps_kill_switch(self):
        rm = make_rm(max_drawdown_pct=0.2)
        rm.update_equity(100_000.0)
        rm.update_equity(70_000.0)   # dispara el kill-switch
        state = rm.to_dict()
        rm2 = make_rm(max_drawdown_pct=0.2)
        rm2.load_dict(state)
        assert rm2.halted and rm2.peak_equity == pytest.approx(100_000.0)

    def test_load_dict_tolerates_missing_keys(self):
        rm = make_rm()
        rm.load_dict({})
        assert not rm.halted and rm.peak_equity is None


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
