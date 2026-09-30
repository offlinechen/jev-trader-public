# Jev Trader / Jev 交易研究

本项目用于检验 Jev 给出的概率能否为 15 分钟 USDT 永续合约价格路径提供有用信息。它是**验证假设的研究流程**，不是已证明盈利的策略，也不是可直接投入实盘的机器人。

This project tests whether Jev's probability estimates contain useful information about 15-minute USDT perpetual price paths. It is a **research pipeline for falsifying hypotheses**, not a proven profitable strategy or a ready-to-run live bot.

项目构建因果市场特征和 1 分钟首触标签，在滚动样本上比较 Jev 与统计基线，并提供隔离的 Freqtrade 信号存储适配器，供诊断性模拟运行。Binance USDⓈ-M 和 OKX USDT 永续合约分别有公开 K 线适配器。支持传入某个交易对，不代表该交易对已通过研究或交易验收。

The project builds causal market features and 1-minute first-touch labels, compares Jev with statistical baselines on walk-forward samples, and provides an isolated Freqtrade signal-store adapter for diagnostic dry-runs. Binance USDⓈ-M and OKX USDT swaps have separate public-candle adapters. Accepting a pair does not mean that pair has passed the research or trading gates.

## 当前状态 / Current status

- Binance 的历史研究和诊断信号路径已经具备，但排序确认关卡尚未完成；校准、净收益回测、持续模拟运行和实盘交易关卡**均未通过**。
  Binance historical research and a diagnostic signal path exist, but the ranking confirmation gate is incomplete; calibration, net-return backtesting, sustained dry-run, and live-trading gates have **not** passed.
- OKX 使用仅含 OHLCV 的 v4 状态契约。一次使用**合成** K 线的付费请求通过了响应格式校验，但不计入研究结果。2026-09-30 已从 Binance 和 OKX 公开端点读取真实 K 线并构建状态；尚未用真实 OKX 行情进行 Jev 推理或端到端模拟运行。
  OKX has an OHLCV-only v4 state contract. One paid request using **synthetic** candles passed response-schema validation, but is excluded from research. On 2026-09-30, real public candles from both Binance and OKX were fetched and converted to states; real-OKX Jev inference and end-to-end dry-run remain unvalidated.
- Freqtrade 示例均启用 `dry_run: true`、不含交易所凭据，不应直接切换为实盘。离线测试通过或模型响应有效，都不意味着可以向交易所下单。
  Freqtrade examples use `dry_run: true` and contain no exchange credentials. Do not switch them directly to live trading. An offline test or a valid model response does not authorize exchange orders.

## 安装与离线验证 / Install and verify offline

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,research]'
cp config/jev.example.yaml config/jev.yaml
env -u JEV_LIVE .venv/bin/python -m pytest -q
```

研究运行前，请在本地 `config/jev.yaml` 中调整历史区间、交易对、网格和**估算**交易成本。以下命令只观察公开市场的已收盘 K 线，不调用 Jev，也不下单：

Before a research run, adjust the historical range, symbol, grid, and **estimated** trading costs in your local `config/jev.yaml`. These commands only observe public closed candles; they neither call Jev nor place orders:

```bash
.venv/bin/jev shadow --exchange binance --symbols BTCUSDT
.venv/bin/jev shadow --exchange okx --symbols BTC/USDT:USDT
```

若返回 `stale`，表示已收盘 K 线超过本项目的一分钟信号时效限制；可在下一根 15m K 线刚收盘时重试。若所在网络或地区无法访问公开端点，命令仍可能失败。不要用合成 K 线代替真实行情验证。

`stale` means the latest closed candle is outside the project's one-minute signal freshness window; retry just after the next 15m close. Public endpoints may still be unavailable from some networks or regions. Do not substitute synthetic candles for real-market validation.

若 macOS 上的 Python 报 `CERTIFICATE_VERIFY_FAILED`，请配置可信的 CA 证书。例如可在虚拟环境安装 `certifi`，再用其证书包运行上述命令；不要关闭 TLS 校验：

If Python on macOS raises `CERTIFICATE_VERIFY_FAILED`, configure a trusted CA bundle. For example, install `certifi` in the virtual environment and use its bundle for the commands above; never disable TLS verification:

```bash
.venv/bin/python -m pip install certifi
SSL_CERT_FILE="$(.venv/bin/python -m certifi)" .venv/bin/jev --config config/jev.example.yaml shadow --exchange okx --symbols BTCUSDT
```

## 配置与付费请求 / Configuration and paid requests

只有在明确要运行付费 Jev 命令时，才将 `.env.example` 复制为 `.env` 并填写 `JEV_API_KEY`。`JEV_BASE_URL`、`JEV_MODEL_ID` 和 `JEV_PROMPT_VERSION` 定义推理契约；本地 YAML 保存网格、预测时长、成本和交易对。Binance 旧版推理路径使用 `v3`；OKX 付费推理要求本地 YAML 使用 `jev-ohlcv-v4`，但真实 OKX K 线尚未验证。`.env`、本地 YAML、报告、数据和交易日志均不得提交到 Git。

Copy `.env.example` to `.env` and fill `JEV_API_KEY` only when deliberately running a paid Jev command. `JEV_BASE_URL`, `JEV_MODEL_ID`, and `JEV_PROMPT_VERSION` define the inference contract; the local YAML holds grid, horizon, costs, and symbols. Binance's legacy inference path uses `v3`; OKX paid inference requires `jev-ohlcv-v4` in the local YAML and remains unvalidated on real OKX candles. Never commit `.env`, local YAML, reports, data, or trading logs.

付费 meter/shadow 运行需要显式设置累计 `--max-requests` 和 `--max-usd` 上限，并写入本地支出账本。单次 smoke 响应既不是交易信号，也不是研究结论。真实交易所凭据不得放入仓库；附带的 Freqtrade 模板仅用于模拟运行，刻意不包含凭据。模板禁用了 API server；若启用它，绝不能沿用公开的占位凭据。

Paid meter/shadow runs require explicit cumulative `--max-requests` and `--max-usd` caps and write a local spend ledger. A smoke response is neither a trading signal nor a research result. Keep real exchange credentials out of this repository; the supplied Freqtrade templates are dry-run only and intentionally contain none. Their API server is disabled; never enable it with the public placeholder credentials.

## 仓库结构 / Repository map

- `src/jev_trader/`：市场数据、特征、标签、模型客户端、基线、评估和诊断性模拟执行。 / Market data, features, labels, model client, baselines, evaluation, and diagnostic paper execution.
- `freqtrade/strategies/JevSignalStore.py`：读取已明确授权的持久化信号，不调用 Jev。 / Reads explicitly authorized persisted signals; it never calls Jev.
- `tests/`：离线回归与集成测试。未单独安装 Freqtrade 时，相关测试会跳过。 / Offline regression and integration tests. Freqtrade-dependent tests skip unless Freqtrade is installed separately.
- `config/*.example.*` 与 `.env.example`：可公开的模板；复制后在本地修改，私有配置已忽略。 / Public templates; copy and edit locally, while private configs are ignored.

交易和模型推理可能造成资金损失或 API 费用。未经独立验证、交易所专属风控和明确的账户级批准，请勿用本代码进行真实资金交易。

Trading and model inference can lose money or incur API charges. Do not use this code for real-money trading without independent validation, exchange-specific risk controls, and explicit account-level approval.

## 许可证 / License

MIT。详见 [LICENSE](LICENSE)。 / MIT. See [LICENSE](LICENSE).
