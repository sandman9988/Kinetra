from click.testing import CliRunner

import scripts.renko_engine as renko_engine


def test_main_forwards_backtest_overrides(monkeypatch):
    captured = {}

    def _fake_stage_backtest(
        symbol,
        months,
        min_omega,
        min_trades,
        stop_bricks_override,
        target_risk_override,
        brick_size_override,
        fliprate_window_override,
        markov_window_override,
        fliprate_threshold_override,
        markov_threshold_override,
    ):
        captured.update(
            {
                "symbol": symbol,
                "months": months,
                "min_omega": min_omega,
                "min_trades": min_trades,
                "stop_bricks_override": stop_bricks_override,
                "target_risk_override": target_risk_override,
                "brick_size_override": brick_size_override,
                "fliprate_window_override": fliprate_window_override,
                "markov_window_override": markov_window_override,
                "fliprate_threshold_override": fliprate_threshold_override,
                "markov_threshold_override": markov_threshold_override,
            }
        )
        return True

    monkeypatch.setattr(renko_engine, "stage_backtest", _fake_stage_backtest)

    runner = CliRunner()
    result = runner.invoke(
        renko_engine.main,
        [
            "XAUUSD",
            "--stage",
            "backtest",
            "--months",
            "2",
            "--min-omega",
            "1.7",
            "--min-trades",
            "40",
            "--stop-bricks",
            "0.6",
            "--target-risk",
            "80",
            "--brick-size",
            "1.2",
            "--fliprate-window",
            "250",
            "--markov-window",
            "260",
            "--fliprate-threshold",
            "0.33",
            "--markov-threshold",
            "0.57",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["symbol"] == "XAUUSD"
    assert captured["months"] == 2
    assert captured["min_omega"] == 1.7
    assert captured["min_trades"] == 40
    assert captured["stop_bricks_override"] == 0.6
    assert captured["target_risk_override"] == 80.0
    assert captured["brick_size_override"] == 1.2
    assert captured["fliprate_window_override"] == 250
    assert captured["markov_window_override"] == 260
    assert captured["fliprate_threshold_override"] == 0.33
    assert captured["markov_threshold_override"] == 0.57


def test_main_forwards_live_dryrun_overrides_and_drift(monkeypatch):
    captured = {}

    def _fake_stage_live_dryrun(
        symbol,
        live_size,
        stop_bricks_override,
        target_risk_override,
        brick_size_override,
        fliprate_window_override,
        markov_window_override,
        fliprate_threshold_override,
        markov_threshold_override,
        drift_adapt,
        drift_opts,
    ):
        captured.update(
            {
                "symbol": symbol,
                "live_size": live_size,
                "stop_bricks_override": stop_bricks_override,
                "target_risk_override": target_risk_override,
                "brick_size_override": brick_size_override,
                "fliprate_window_override": fliprate_window_override,
                "markov_window_override": markov_window_override,
                "fliprate_threshold_override": fliprate_threshold_override,
                "markov_threshold_override": markov_threshold_override,
                "drift_adapt": drift_adapt,
                "drift_opts": drift_opts,
            }
        )
        return True

    monkeypatch.setattr(renko_engine, "stage_live_dryrun", _fake_stage_live_dryrun)

    runner = CliRunner()
    result = runner.invoke(
        renko_engine.main,
        [
            "XAUUSD",
            "--stage",
            "live",
            "--dry-run",
            "--live-size",
            "small",
            "--stop-bricks",
            "0.7",
            "--target-risk",
            "90",
            "--brick-size",
            "1.1",
            "--drift-adapt",
            "--adapt-every-bars",
            "55",
            "--adapt-min-trades",
            "7",
            "--adapt-low-omega",
            "0.9",
            "--adapt-high-omega",
            "1.9",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["symbol"] == "XAUUSD"
    assert captured["live_size"] == "small"
    assert captured["stop_bricks_override"] == 0.7
    assert captured["target_risk_override"] == 90.0
    assert captured["brick_size_override"] == 1.1
    assert captured["drift_adapt"] is True
    assert captured["drift_opts"]["every_bars"] == 55
    assert captured["drift_opts"]["min_trades"] == 7
    assert captured["drift_opts"]["low_omega"] == 0.9
    assert captured["drift_opts"]["high_omega"] == 1.9


def test_main_forwards_live_real_preflight_and_ack(monkeypatch):
    captured = {}

    def _fake_stage_live_real(
        symbol,
        live_size,
        preflight_test_order,
        ack_live,
        preflight_lots,
        stop_bricks_override,
        target_risk_override,
        brick_size_override,
        fliprate_window_override,
        markov_window_override,
        fliprate_threshold_override,
        markov_threshold_override,
        drift_adapt,
        drift_opts,
    ):
        captured.update(
            {
                "symbol": symbol,
                "live_size": live_size,
                "preflight_test_order": preflight_test_order,
                "ack_live": ack_live,
                "preflight_lots": preflight_lots,
                "stop_bricks_override": stop_bricks_override,
                "target_risk_override": target_risk_override,
                "drift_adapt": drift_adapt,
                "drift_opts": drift_opts,
            }
        )
        return True

    monkeypatch.setattr(renko_engine, "stage_live_real", _fake_stage_live_real)

    runner = CliRunner()
    result = runner.invoke(
        renko_engine.main,
        [
            "XAUUSD",
            "--stage",
            "live",
            "--live-size",
            "micro",
            "--preflight-test-order",
            "--preflight-lots",
            "0.01",
            "--ack-live",
            "I_UNDERSTAND_LIVE_RISK",
            "--stop-bricks",
            "0.5",
            "--target-risk",
            "100",
            "--drift-adapt",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["symbol"] == "XAUUSD"
    assert captured["live_size"] == "micro"
    assert captured["preflight_test_order"] is True
    assert captured["ack_live"] == "I_UNDERSTAND_LIVE_RISK"
    assert captured["preflight_lots"] == 0.01
    assert captured["stop_bricks_override"] == 0.5
    assert captured["target_risk_override"] == 100.0
    assert captured["drift_adapt"] is True
