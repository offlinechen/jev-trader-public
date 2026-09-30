# Jev Trader

Research code for testing whether Jev's probability estimates contain useful information about 15-minute USDT perpetual price paths. It is a **falsification pipeline**, not a profitable trading system or a ready-to-run live bot.

The project builds causal market features and 1-minute first-touch labels, compares Jev with statistical baselines on walk-forward samples, and has an isolated Freqtrade signal-store adapter for diagnostic dry-runs. Binance USDⓈ-M and OKX USDT swaps have separate public-candle adapters. Code accepting a pair does not mean that pair has passed the research or trading gates.

## Current status

- Binance historical research and a diagnostic signal path exist. The ranking confirmation gate remains incomplete; calibration, net-return backtesting, sustained dry-run, and live-trading gates have **not** passed.
- OKX has an OHLCV-only v4 state contract. One paid request using **synthetic** candles passed response-schema validation; it is excluded from research. Real OKX market-data access returned HTTP 403 on the development host, so no real-data OKX inference or end-to-end dry-run has been validated.
- Freqtrade examples are `dry_run: true`, use blank exchange credentials, and must not be switched to live trading. No exchange order should be inferred from an offline test or a valid model response.

## Install and run offline tests

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,research]'
cp config/jev.example.yaml config/jev.yaml
env -u JEV_LIVE .venv/bin/python -m pytest -q
```

Change the historical date range, symbol, grid, and **estimated** trading costs in your local `config/jev.yaml` before a research run. To observe public closed candles without invoking Jev or placing orders:

```bash
.venv/bin/jev shadow --exchange binance --symbols BTCUSDT
.venv/bin/jev shadow --exchange okx --symbols BTC/USDT:USDT
```

The OKX command may fail if the public endpoint is unavailable from your network or region. Do not substitute synthetic candles for real-market validation.

## Configuration and paid requests

Copy `.env.example` to `.env` and fill `JEV_API_KEY` only if you deliberately run a paid Jev command. `JEV_BASE_URL`, `JEV_MODEL_ID`, and `JEV_PROMPT_VERSION` identify the inference contract; the local YAML holds grid, horizon, costs, and symbol settings. Binance's legacy inference path uses `v3`; OKX paid inference requires `jev-ohlcv-v4` in the local YAML and remains unvalidated on real OKX candles. Neither `.env` nor local YAML, reports, data, or trading logs belong in Git.

Paid meter/shadow runs require explicit cumulative `--max-requests` and `--max-usd` caps and write a local spend ledger. A smoke response is not a trading signal or a research result. Keep real exchange credentials out of this repository; the supplied Freqtrade templates are dry-run only and intentionally contain none. Their API server is disabled; the public placeholder credentials must never be used with it enabled.

## Repository map

- `src/jev_trader/`: market data, features, labels, model client, baselines, evaluation, and diagnostic paper execution.
- `freqtrade/strategies/JevSignalStore.py`: reads explicitly authorized persisted signals; it never calls Jev.
- `tests/`: offline regression and integration tests. Freqtrade-dependent tests skip unless Freqtrade is installed separately.
- `config/*.example.*` and `.env.example`: public templates. Copy and edit locally; local configs are ignored.

Trading and model inference can lose money or incur API charges. Do not use this code for real-money trading without independent validation, exchange-specific risk controls, and explicit account-level approval.

## License

MIT. See [LICENSE](LICENSE).
