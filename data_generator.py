import os
import zlib

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import db

SEED_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed_cards.csv")


class MockDataGenerator:
    """Generates synthetic price histories. It is now only used to FILL the database
    (see seed_database below) - the app itself reads prices from db.py."""

    def __init__(self, seed=42):
        self.seed = seed

    def generate_history(self, card_id, days=180, base_price=None):
        # Per-card RNG: deterministic for a given (seed, card_id) and independent between cards.
        # (Python's built-in hash() changes every run, so a stable crc32 is used instead.)
        rng = np.random.default_rng(self.seed + zlib.crc32(card_id.encode()))
        dates = [datetime.today().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=i)
                 for i in range(days, 0, -1)]

        if base_price is None:
            base_price = 50.0 + (zlib.crc32(card_id.encode()) % 200)

        # Generate synthetic price path with random shocks
        returns = rng.normal(loc=0.0005, scale=0.025, size=days)

        # Inject occasional variance spikes (buyout or reprint crashes)
        shock_indices = rng.choice(days, size=3, replace=False)
        returns[shock_indices[0]] = 0.15   # Hype buyout spike (+15%)
        returns[shock_indices[1]] = -0.12  # Reprint crash (-12%)
        returns[shock_indices[2]] = 0.08   # Creator spotlight (+8%)

        price_series = base_price * np.cumprod(1 + returns)

        # Trading volume correlated with absolute price shifts
        volumes = rng.poisson(lam=120, size=days) + (np.abs(returns) * 1000).astype(int)

        # PSA 10 Graded Premium multiplier (typically 3.5x to 8x raw price)
        psa_multiplier = rng.uniform(4.0, 6.5, size=days)

        df = pd.DataFrame({
            "Date": dates,
            "Market Price ($)": np.round(price_series, 2),
            "Daily Trading Volume": volumes,
            "PSA 10 Multiplier": np.round(psa_multiplier, 2),
            "Catalyst Flag": [i in shock_indices for i in range(days)]
        })
        df.set_index("Date", inplace=True)
        return df


def seed_database(days=180, seed=42, csv_path=SEED_CSV, reset=True):
    """(Re)build the database from seed_cards.csv with made-up price histories.
    To add a card, add a row to seed_cards.csv and run:  python data_generator.py"""
    if reset:
        db.reset_db()
    else:
        db.init_db()

    gen = MockDataGenerator(seed=seed)
    cards = pd.read_csv(csv_path)
    for _, c in cards.iterrows():
        db.upsert_card(c["card_id"], c["name"], c["set_name"], c["rarity"], c["base_price"])
        db.replace_price_history(c["card_id"], gen.generate_history(c["card_id"], days, c["base_price"]))
    return len(cards)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build pokequant.db with made-up card data")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    n = seed_database(a.days, a.seed)
    print(f"Seeded {n} cards x {a.days} days into {db.DB_PATH}")
