from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config, data, labels as lab
from .features import build_features


def cmd_download(args, cfg):
    for symbol in cfg["symbols"]:
        binance = config.to_binance(symbol)
        for tf in args.timeframes:
            print(f"{binance} {tf} {cfg['history']['start']}..{cfg['history']['end']}")
            out = data.download(
                binance, tf, cfg["history"]["start"], cfg["history"]["end"],
                cfg["data_dir"],
            )
            df = pd.read_parquet(out)
            missing = data.gaps(df, tf)
            total = missing["missing"].sum() if len(missing) else 0
            span = (df["ts"].iloc[-1] - df["ts"].iloc[0]) / data.TF_MS[tf] + 1
            print(
                f"  {len(df):,} bars -> {out}  "
                f"({total:,} missing, {100 * len(df) / span:.3f}% coverage)"
            )


def cmd_features(args, cfg):
    for symbol in cfg["symbols"]:
        binance = config.to_binance(symbol)
        df = data.load(binance, cfg["timeframe"], cfg["data_dir"])
        f = build_features(df)
        f.insert(1, "symbol", symbol)
        out = Path(cfg["data_dir"]) / "features.parquet"
        f.to_parquet(out, index=False)
        print(f"{symbol}: {len(f):,} rows x {f.shape[1] - 2} features -> {out}")


def cmd_labels(args, cfg):
    grid = cfg["grid"]
    for symbol in cfg["symbols"]:
        binance = config.to_binance(symbol)
        df15 = data.load(binance, cfg["timeframe"], cfg["data_dir"])
        df1 = data.load(binance, cfg["label_timeframe"], cfg["data_dir"])
        out = lab.build_labels(df15, df1, grid["levels"], grid["horizon_bars"])
        out.insert(1, "symbol", symbol)
        path = Path(cfg["data_dir"]) / "labels.parquet"
        out.to_parquet(path, index=False)
        print(
            f"{symbol}: {len(out):,} labelled bars "
            f"({len(df15) - len(out)} dropped at the right edge) -> {path}"
        )

        s = lab.summary(out, grid)
        print("\n-- T1.7 per-cell rates --")
        print(s.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(
            f"\nambiguity: mean {s.ambiguous_pct.mean():.4f}%  "
            f"max {s.ambiguous_pct.max():.4f}% "
            f"(tp={s.loc[s.ambiguous_pct.idxmax(), 'tp']}, "
            f"sl={s.loc[s.ambiguous_pct.idxmax(), 'sl']})"
        )


def cmd_baselines(args, cfg):
    from . import baseline, report
    F = pd.read_parquet(Path(cfg["data_dir"]) / "features.parquet")
    L = pd.read_parquet(Path(cfg["data_dir"]) / "labels.parquet")
    baseline.run(F, L, cfg["grid"], cfg["data_dir"],
                 train_months=args.train_months, test_months=args.test_months)
    s = report.summary(cfg["data_dir"])
    print("\n-- within-cell (headline) --")
    print(s["within_cell"].round(4).to_string())
    print(f"\nA_base_direction = {s['A_base_direction']:.4f}   A_vol = {s['A_vol']:.4f}")
    print(f"B_clim = {s['B_clim']:.4f}   B_lgbm = {s['B_lgbm']:.4f}   B_logit = {s['B_logit']:.4f}")


def cmd_api_smoke(args, cfg):
    from .jev import JevClient, JevSettings

    settings = JevSettings.from_config(cfg)
    result = JevClient(settings).decide(
        {"test": "api contract smoke", "version": 1},
        {"is_smoke": {"type": "noul", "instructions": "Is this a smoke test request?"}},
        use_cache=not args.no_cache,
    )
    print(
        f"model={result.model} answer={result.answers['is_smoke']:.4f} "
        f"latency_ms={result.latency_ms:.1f} cached={result.cached} "
        f"in_tokens={result.usage['input_tokens']} out_tokens={result.usage['output_tokens']}"
    )


def cmd_meter(args, cfg):
    from .meter import run

    report = run(
        n=args.n,
        seed=args.seed,
        workers=args.workers,
        use_cache=not args.no_cache,
        max_requests=args.max_requests,
        max_usd=args.max_usd,
        est_cost_per_request=args.est_cost,
        log_every=args.log_every,
        health_min_samples=args.health_min_samples,
        health_window=args.health_window,
        max_invalid_rate=args.max_invalid_rate,
        allow_health_override=args.allow_health_override,
        contiguous=args.contiguous,
        prompt_version=args.prompt_version,
        choice_barriers=args.choice_barriers,
    )
    print(f"report -> {report}")


def cmd_audit_spend(args, cfg):
    import os

    from .audit import run
    from .jev import _read_dotenv

    root = Path(__file__).resolve().parents[2]
    env = {**_read_dotenv(root / ".env"), **os.environ}   # 不打印密钥 / Never print the key.
    text, _ = run(
        root, env.get("JEV_API_KEY"), env.get("JEV_BASE_URL"),
        env.get("OPENROUTER_MANAGEMENT_KEY"),
        query_provider=not args.offline,
    )
    out = root / "docs" / "spend_audit.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"written -> {out}")


def cmd_selective(args, cfg):
    from .selective import run_report

    out = "docs/selective_report.md" if args.model == "lgbm" else f"docs/selective_report_{args.model}.md"
    compare = _default_compare(args.model) if args.compare is None else (args.compare or None)
    if compare == args.model:
        raise ValueError("--compare must differ from --model")
    print(f"report -> {run_report(cfg['data_dir'], cfg, model=args.model, compare=compare, out=out)}")


def _default_compare(model: str) -> str:
    return {"lgbm": "jev", "jev": "lgbm", "stack": "jev"}[model]


def cmd_policies(args, cfg):
    from .policy import run_report

    out = "docs/policy_report.md" if args.model == "lgbm" else f"docs/policy_report_{args.model}.md"
    print(f"report -> {run_report(cfg['data_dir'], cfg, model=args.model, out=out)}")


def cmd_backtest(args, cfg):
    from .backtest import run_file

    trades, decisions, report = run_file(args.predictions, cfg)
    print(f"trades -> {trades}")
    print(f"decisions -> {decisions}")
    print(f"report -> {report}")


def cmd_g3a(args, cfg):
    from .g3a import run

    print(f"report -> {run(args.predictions, cfg['data_dir'])}")


def cmd_confirmation(args, cfg):
    from .confirmation import run

    print(f"report -> {run(args.predictions, cfg['data_dir'], args.out, args.bootstrap)}")


def cmd_shadow(args, cfg):
    import json
    import time

    from .shadow import run_once

    symbols = args.symbols or cfg["symbols"]
    if args.live_jev and (args.max_requests is None or args.max_usd is None):
        raise ValueError("--live-jev requires cumulative --max-requests and --max-usd")
    if args.dry_run_trades and not args.live_jev:
        raise ValueError("--dry-run-trades requires --live-jev")
    while True:
        if args.watch:
            # 下根 15m 收盘后运行；run_once 拒绝过期 K 线。 / Run after close; reject stale bars.
            now = time.time()
            time.sleep(max(0, (int(now // 900) + 1) * 900 + 2 - now))
        for row in run_once(
            symbols, cfg, exchange=args.exchange, live_jev=args.live_jev,
            dry_run_trades=args.dry_run_trades,
            max_requests=args.max_requests, max_usd=args.max_usd,
        ):
            print(json.dumps(row, sort_keys=True), flush=True)
        if not args.watch:
            break


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="jev")
    p.add_argument("--config", default=config.DEFAULT)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch OHLCV archives")
    d.add_argument("--timeframes", nargs="+", default=["15m", "1m"])
    d.set_defaults(fn=cmd_download)

    f = sub.add_parser("features", help="build the feature frame")
    f.set_defaults(fn=cmd_features)

    b = sub.add_parser("labels", help="build first-touch labels + T1.7 report")
    b.set_defaults(fn=cmd_labels)

    m = sub.add_parser("baselines", help="purged walk-forward baselines + OOF")
    m.add_argument("--train-months", type=int, default=6)
    m.add_argument("--test-months", type=int, default=1)
    m.set_defaults(fn=cmd_baselines)

    a = sub.add_parser("api-smoke", help="one live Jev Decisions API contract request")
    a.add_argument("--no-cache", action="store_true", help="force a real request")
    a.set_defaults(fn=cmd_api_smoke)

    t = sub.add_parser("meter", help="budget-guarded live Jev run over stratified bars")
    t.add_argument("--n", type=int, required=True)
    t.add_argument("--seed", type=int, default=20260921)
    t.add_argument("--workers", type=int, default=1)
    t.add_argument("--max-requests", type=int, required=True,
                   help="hard cap on HTTP attempts, retries included")
    t.add_argument("--max-usd", type=float, required=True,
                   help="hard cap on committed spend; stops locally")
    t.add_argument("--est-cost", type=float, default=0.000284,
                   help="$/request for pre-flight and unreported-cost charging")
    t.add_argument("--log-every", type=int, default=100)
    t.add_argument("--health-min-samples", type=int, default=10,
                   help="minimum completed bars before schema-health stopping is enabled")
    t.add_argument("--health-window", type=int, default=20,
                   help="recent completed-bar window for schema-health stopping")
    t.add_argument("--max-invalid-rate", type=float, default=0.25,
                   help="stop after the recent invalid-schema fraction exceeds this rate")
    t.add_argument("--allow-health-override", action="store_true",
                   help="explicitly authorize a new tranche after a schema-health stop")
    t.add_argument("--no-cache", action="store_true")
    t.add_argument("--contiguous", action="store_true",
                   help="use the latest n complete consecutive labelled bars for a diagnostic replay")
    t.add_argument("--prompt-version", choices=("v3", "jev-ohlcv-v4"),
                   help="override the prompt for this run without changing config")
    t.add_argument("--choice-barriers", action="store_true",
                   help="diagnostic mutually exclusive TP/SL/timeout questions")
    t.set_defaults(fn=cmd_meter)

    s = sub.add_parser("audit-spend", help="reconcile Jev spend against OpenRouter")
    s.add_argument("--offline", action="store_true",
                   help="read local credentials for leak checks but skip provider queries")
    s.set_defaults(fn=cmd_audit_spend)

    v = sub.add_parser("selective", help="coverage/EV/regime/agreement report on OOF")
    v.add_argument("--model", choices=("lgbm", "jev", "stack"), default="lgbm")
    v.add_argument("--compare", default=None,
                   help="second registered model for agreement; pass an empty value for none")
    v.set_defaults(fn=cmd_selective)

    q = sub.add_parser("policies", help="fixed / argmax / robust / softmax / multi-TP comparison")
    q.add_argument("--model", choices=("lgbm", "jev", "stack"), default="lgbm")
    q.set_defaults(fn=cmd_policies)

    r = sub.add_parser("backtest", help="replay Jev predictions with TP/SL and costs")
    r.add_argument("--predictions", default="data/metering_100.parquet")
    r.set_defaults(fn=cmd_backtest)

    g = sub.add_parser("g3a", help="evaluate raw Jev ranking and incremental value")
    g.add_argument("--predictions", default="data/metering_2000.parquet")
    g.set_defaults(fn=cmd_g3a)

    c = sub.add_parser("confirmation", help="matched, fold-aware G3b ranking confirmation")
    c.add_argument("--predictions", required=True,
                   help="complete run-identified 10k+ meter parquet; legacy partial files are rejected")
    c.add_argument("--out", default="docs/confirmation_report.md")
    c.add_argument("--bootstrap", type=int, default=400)
    c.set_defaults(fn=cmd_confirmation)

    sh = sub.add_parser("shadow", help="real closed candles to Jev observations; optional local paper simulation")
    sh.add_argument("--exchange", choices=("binance", "okx"), default="binance")
    sh.add_argument("--symbols", nargs="+", help="USDT-M pairs; accepts ETHUSDT or ETH/USDT:USDT")
    sh.add_argument("--watch", action="store_true", help="observe every new 15m close")
    sh.add_argument("--live-jev", action="store_true", help="explicitly allow paid Jev requests")
    sh.add_argument("--dry-run-trades", action="store_true",
                    help="simulate fixed diagnostic orders locally; never sends exchange orders")
    sh.add_argument("--max-requests", type=int, help="cumulative HTTP attempt cap for this shadow identity")
    sh.add_argument("--max-usd", type=float, help="cumulative spend cap for this shadow identity")
    sh.set_defaults(fn=cmd_shadow)

    args = p.parse_args(argv)
    if args.cmd == "confirmation" and args.bootstrap < 400:
        p.error("confirmation --bootstrap must be at least 400 for the frozen G3b gate")
    return args.fn(args, config.load(args.config)) or 0


if __name__ == "__main__":
    raise SystemExit(main())
