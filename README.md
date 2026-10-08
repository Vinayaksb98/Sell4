# NIFTY 500 Core + Extraordinary Sell Scanner

Streamlit app for NIFTY 500 selling-side screening.

- No market-cap filter.
- 1 Day / 1 Week / 1 Hour can be scanned separately.
- Core support: prior low with Stochastic <20 and MACD <0.
- Breakdown Confirmed: later candle closes below support with Stochastic <20 and MACD <0, and current CMP remains below support with MACD <0.
- Near Breakdown: CMP is above support but within 3%, with bearish momentum.
- Extraordinary markers are informational only and do not reject a Core candidate.
- CMP is deliberately not cached and is refreshed on each scan.
- Table order is Stock | CMP | Support Price | Breakdown Status.

Run:
`streamlit run app.py`
