# XAUUSD ML strategy for FTMO

A machine-learning strategy for gold (XAUUSD) built around FTMO's challenge rules:
gradient-boosted classifiers, purged walk-forward validation, a cost-aware bar-by-bar
backtester, a Monte-Carlo-style FTMO challenge simulator, and a MetaTrader 5 live trader.

> **Read this first.** This repo gives you a sound *research and execution framework*, not
> a proven edge. It has only been tested on synthetic data, because no real market data
> was available where it was built. Run it on your own FTMO/MT5 history. Only trade it if the
> **out-of-sample** results hold up across thresholds and time periods. Then forward-test on an
> FTMO Free Trial before you pay for a challenge. Most ML trading models don't beat costs.

## How it works

| Step | Module | What it does |
|---|---|---|
| Data | `data.py` | Loads MT5 "Export bars" CSVs or generic OHLC CSVs, or pulls history from a running MT5 terminal |
| Features | `features.py` | ~45 causal features: multi-horizon returns in ATR units, volatility regime, RSI/MACD/EMA distances, Bollinger %b, range position, candle shape, tick-volume, previous-day levels, session/hour |
| Labels | `labels.py` | Triple barrier that mirrors the real trade: enter next bar open, TP = 2×ATR, SL = 1×ATR, time exit after 24 bars, spread included, same-bar TP+SL counted as a loss |
| Model | `model.py` | Two `HistGradientBoostingClassifier`s: P(long TP before SL) and P(short TP before SL). Trade only when P ≥ threshold *and* beats the other side by `min_edge` |
| Validation | `model.walk_forward` | Rolling ~2-year training window, retrains every ~month, purges training rows whose label window overlaps the test block |
| Risk | `risk.py` | FTMO 5% daily / 10% max loss, plus stricter internal stops (2.5% daily, 7% total), 0.5% risk per trade, halve risk after 2 losses, sizing that never lets a stop-out get within 0.5% of an FTMO limit |
| Backtest | `backtest.py` | Next-bar-open execution, bid/ask, slippage, commission, gaps through stops, intrabar worst-case equity checked against FTMO limits, Friday flatten |
| Challenge sim | `backtest.simulate_challenges` | Starts a fresh $100k challenge every 7 days of the out-of-sample period → pass / fail / timeout rates |
| Live | `live.py` | MT5 loop: on each closed bar predict → size → send order with broker-side SL/TP; a guard runs every 5 s that flattens on equity limits, time stop and weekend |

## Quick start

```bash
cd xauusd_ftmo
pip install -r requirements.txt
python -m pytest -q tests            # 7 tests: no-lookahead, labels, sizing, FTMO breach, model sanity

# 1. Get data. On Windows with MT5 logged in to FTMO:
pip install MetaTrader5
python -m ftmo_ml.cli fetch-mt5 --bars 60000 --out data/xauusd_h1.csv
#    ...or export from MT5 (View → Symbols → Bars → XAUUSD H1 → Export) and use that CSV.
#    No data handy? `python -m ftmo_ml.cli synth` makes a synthetic file for smoke tests only.

# 2. Research: walk-forward out-of-sample backtest + FTMO challenge simulation
python -m ftmo_ml.cli backtest --data data/xauusd_h1.csv
#    -> reports/summary.json, trades.csv, equity.csv, challenges.csv, oos_probabilities.csv

# 3. Fit the production model on the most recent window
python -m ftmo_ml.cli train --data data/xauusd_h1.csv

# 4. Forward test on an FTMO Free Trial: dry run first (logs only), then real orders
python -m ftmo_ml.cli live
python -m ftmo_ml.cli live --live
```

You need at least ~13k H1 bars (about 2+ years) before walk-forward starts. 8–10 years gives you
a meaningful out-of-sample period.

## Reading the results

`summary.json` contains:

- **classification**: out-of-sample AUC and the hit rate of the top decile, versus the base rate.
  An AUC of ~0.50 means no edge. 0.53–0.56 is typical of a real but small edge.
- **threshold_sweep**: results at several probability thresholds. If only one threshold makes
  money, treat it as overfitting.
- **backtest**: return, max drawdown, worst daily loss (as a % of initial balance, the way FTMO
  measures it), profit factor, win rate, average R and exit reasons.
- **challenges**: pass / fail / timeout rates for challenges started every week. This tells you
  most directly whether it's worth paying for a challenge.

With TP 2R / SL 1R the break-even win rate is ~34% before costs. The model needs to pick trades
clearly above that.

## Things to tune (config.yaml)

- `costs`: set spread, commission and contract size to match **your** FTMO MT5 symbol
  specification. Live trading reads contract size and lot step from the broker automatically.
- `labels.tp_atr / sl_atr / horizon_bars`: the trade structure. If you change these, the model
  must be retrained.
- `model.prob_threshold / min_edge`: trade frequency versus quality.
- `ftmo.profit_target_pct`: 10 for phase 1, 5 for phase 2. On a funded account, set it very high
  so trading never stops.
- `risk.*`: 0.5% per trade means ~10 straight losses before the internal 7% stop. Raising it
  makes passing faster but failing far more likely.

## Assumptions and limits

- Prices are MT5 bid bars. Intrabar order is unknown, so if a bar touches both SL and TP the
  backtest assumes the stop was hit first.
- The day boundary is the data's timestamp date, which on FTMO servers matches the FTMO (CE(S)T)
  daily reset. The day-start level is `max(balance, equity)`, the stricter reading of the
  daily-loss rule.
- Once the target is hit, the simulator assumes one 0.01-lot trade per day until the minimum
  number of trading days is reached.
- There is no news filter. FTMO restricts trading around high-impact news on some funded
  accounts, so check your account type and add a calendar filter if needed.
- FTMO's rules change. Check the current rules on their site, including their rules on EAs and
  prohibited strategies.
- This is not financial advice. You can lose your challenge fee and, on live accounts, money.
