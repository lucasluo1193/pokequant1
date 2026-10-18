# PokeQuant

A Pokémon card quantitative trading dashboard built with Streamlit.

## Features

- Search and browse the local card database from the home page.
- Open any card to view its price history and quantitative metrics.
- Get a rules-based **BUY / HOLD / SELL** signal using:
  - short-term vs long-term moving-average trend
  - 30-day momentum
  - Monte Carlo expected return
  - volatility/risk adjustment
- Load and update real TCGplayer market prices through the Pokémon TCG API.
- Import historical price data from CSV.
- View Monte Carlo / jump-diffusion forecast paths.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The database is built automatically on first run if it is missing.

## Real prices

Prices come from the Pokémon TCG API (`pokemontcg.io`), which includes TCGplayer market prices.

The API normally returns today's price, so historical data builds up as prices are refreshed. You can also import historical prices from CSV.

## Quant signal

The trading signal is a rules-based simulation for the app rather than a real trading system:

- **BUY**: score >= +25
- **HOLD**: score between -24 and +24
- **SELL**: score <= -25

The score combines trend, momentum, expected Monte Carlo return, and volatility.

Analysis uses calendar days and 365-day annualization. Missing dates are
forward-filled from the last observation for analysis only; stored prices stay
unchanged. Sparse observations can therefore understate day-to-day volatility.
Both forecasts use daily log-return estimates, with one simulated step per
future calendar day. Confidence is a heuristic, not calibrated accuracy.

Print variants have separate tracked card IDs. Existing databases are migrated
without changing history. Legacy API-linked cards without a recorded variant are
skipped during refresh: save a new card with the correct variant to start a
separate history. Cards with real history cannot be reassigned to a different
API identity or variant.

CSV timestamps are normalized to UTC calendar dates. For multiple observations
on one date, the latest timestamp wins (the last input row wins ties). Prices
must be finite and positive; optional volumes must be nonnegative whole numbers.
Replacing demo history and importing new rows happen in one transaction.

Run regression checks with `python -m unittest test_regressions -v`.

## Files

- `app.py` — main Streamlit interface
- `db.py` — SQLite database access
- `price_sources.py` — Pokémon TCG API and CSV price loading
- `price_ui.py` — real-price management screen
- `math_model.py` — volatility, Monte Carlo, forecasting, and BUY/HOLD/SELL model
- `data_generator.py` — starter database generator
- `seed_cards.csv` — starter cards
- `pokequant.db` — SQLite database
- `assets/database/` — local card assets
