"""
db.py - SQLite storage for PokeQuant.

This is the only module that talks to the database. Everything else (scanner,
app, data_generator) goes through the functions below, so swapping SQLite for
Postgres or a real price feed later only means changing this file.

Tables
  cards   one row per card (id slug, name, set, rarity, ...)
  prices  one row per card per day (price, volume, PSA 10 multiplier, catalyst flag)
"""
import os
import sqlite3
import math
from contextlib import contextmanager

import pandas as pd

DB_PATH = os.environ.get(
    "POKEQUANT_DB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "pokequant.db"),
)

CARDS_DDL = """
CREATE TABLE IF NOT EXISTS cards (
    card_id     TEXT PRIMARY KEY,          -- slug, e.g. 'charizard_holo' (matches reference image filename)
    name        TEXT NOT NULL,
    set_name    TEXT NOT NULL,
    rarity      TEXT NOT NULL,
    base_price  REAL NOT NULL,             -- starting price used when generating mock history
    api_id      TEXT,                      -- pokemontcg.io id
    variant     TEXT
);
"""

PRICES_DDL = """
CREATE TABLE IF NOT EXISTS prices (
    card_id          TEXT NOT NULL REFERENCES cards(card_id) ON DELETE CASCADE,
    price_date       TEXT NOT NULL,        -- ISO date
    market_price     REAL NOT NULL,
    volume           INTEGER,              -- NULL when the source doesn't provide it
    psa10_multiplier REAL,                 -- NULL when the source doesn't provide it
    catalyst_flag    INTEGER NOT NULL DEFAULT 0,
    source           TEXT NOT NULL DEFAULT 'mock',   -- 'mock' | 'tcgplayer' | 'csv' | 'manual'
    PRIMARY KEY (card_id, price_date)
);
"""

SCHEMA = CARDS_DDL + PRICES_DDL + """
CREATE INDEX IF NOT EXISTS idx_prices_card_date ON prices(card_id, price_date);
"""


def _columns(conn, table):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def _migrate(conn):
    """Upgrade a database created by an older version (adds api_id / source columns and
    makes volume / psa10_multiplier optional). Safe to run repeatedly; keeps all data."""
    cards_cols = _columns(conn, "cards")
    if cards_cols and "api_id" not in cards_cols:
        conn.execute("ALTER TABLE cards ADD COLUMN api_id TEXT")
    if cards_cols and "variant" not in cards_cols:
        conn.execute("ALTER TABLE cards ADD COLUMN variant TEXT")
    price_cols = _columns(conn, "prices")
    if price_cols and "source" not in price_cols:
        conn.execute("ALTER TABLE prices RENAME TO prices_old")
        conn.executescript(PRICES_DDL)
        conn.execute(
            """INSERT INTO prices (card_id, price_date, market_price, volume, psa10_multiplier,
                                   catalyst_flag, source)
               SELECT card_id, price_date, market_price, volume, psa10_multiplier, catalyst_flag, 'mock'
               FROM prices_old""")
        conn.execute("DROP TABLE prices_old")


_checked_paths = set()


def _ensure_schema():
    """Create / upgrade the tables the first time each database file is opened in this process,
    so every entry point (app, scripts, importers) always sees the current schema."""
    if DB_PATH in _checked_paths:
        return
    conn = sqlite3.connect(DB_PATH)
    try:
        _migrate(conn)
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
    _checked_paths.add(DB_PATH)


@contextmanager
def connect():
    _ensure_schema()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist yet (and upgrade an older database file)."""
    with connect():
        pass


def initialize_database():
    """Seed only a missing database; existing empty databases belong to the user."""
    is_new = not os.path.exists(DB_PATH)
    init_db()
    if is_new:
        from data_generator import seed_database
        seed_database(reset=False)


def reset_db():
    """Delete the database file and recreate empty tables."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    _checked_paths.discard(DB_PATH)
    init_db()


def has_data() -> bool:
    if not os.path.exists(DB_PATH):
        return False
    with connect() as conn:
        n_cards = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
        n_prices = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    return n_cards > 0 and n_prices > 0


# ----------------------------------------------------------------- writes ----
def upsert_card(card_id, name, set_name, rarity, base_price, api_id=None, variant=None):
    with connect() as conn:
        conn.execute(
            """INSERT INTO cards (card_id, name, set_name, rarity, base_price, api_id, variant)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(card_id) DO UPDATE SET
                 name=excluded.name, set_name=excluded.set_name,
                 rarity=excluded.rarity, base_price=excluded.base_price,
                 api_id=COALESCE(excluded.api_id, cards.api_id),
                 variant=COALESCE(excluded.variant, cards.variant)""",
            (card_id, name, set_name, rarity, float(base_price), api_id, variant),
        )


def link_real_card(card_id, api_id, name, set_name, rarity, variant=None):
    """Turn an existing (mock) card into a real-price card: update its details, store its
    pokemontcg.io id and delete its MOCK price rows so fake and real data never mix."""
    with connect() as conn:
        existing = conn.execute("SELECT api_id, variant FROM cards WHERE card_id=?", (card_id,)).fetchone()
        if existing is None:
            raise ValueError("Card not found.")
        has_real = conn.execute("SELECT 1 FROM prices WHERE card_id=? AND source!='mock' LIMIT 1", (card_id,)).fetchone()
        if has_real and (existing[0] != api_id or existing[1] != variant):
            raise ValueError("This card has real history for a different or unconfirmed identity/variant. Save as a new card instead.")
        conn.execute("UPDATE cards SET api_id=?, name=?, set_name=?, rarity=?, variant=? WHERE card_id=?",
                     (api_id, name, set_name, rarity, variant, card_id))
        conn.execute("DELETE FROM prices WHERE card_id=? AND source='mock'", (card_id,))


def replace_price_history(card_id, df: pd.DataFrame):
    """Replace a card's whole price history with `df` (index = Date, columns as produced by
    MockDataGenerator.generate_history)."""
    rows = [
        (card_id, ts.strftime("%Y-%m-%d"), float(r["Market Price ($)"]),
         int(r["Daily Trading Volume"]), float(r["PSA 10 Multiplier"]), int(bool(r["Catalyst Flag"])), "mock")
        for ts, r in df.iterrows()
    ]
    with connect() as conn:
        conn.execute("DELETE FROM prices WHERE card_id = ?", (card_id,))
        conn.executemany(
            """INSERT INTO prices (card_id, price_date, market_price, volume, psa10_multiplier,
                                   catalyst_flag, source) VALUES (?,?,?,?,?,?,?)""", rows)


def add_price(card_id, price_date, market_price, volume=None, psa10_multiplier=None,
              catalyst_flag=False, source="manual"):
    """Insert/update a single day's observation. One row per card per day: recording again on
    the same date overwrites that day. `source` tags where the number came from."""
    with connect() as conn:
        conn.execute(
            """INSERT INTO prices (card_id, price_date, market_price, volume, psa10_multiplier,
                                   catalyst_flag, source)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(card_id, price_date) DO UPDATE SET
                 market_price=excluded.market_price, volume=excluded.volume,
                 psa10_multiplier=excluded.psa10_multiplier, catalyst_flag=excluded.catalyst_flag,
                 source=excluded.source""",
            (card_id, str(price_date)[:10], float(market_price),
             None if volume is None else int(volume),
             None if psa10_multiplier is None else float(psa10_multiplier),
             int(bool(catalyst_flag)), source),
        )


def add_prices_bulk(card_id, rows, source="csv", replace_mock=False):
    """rows: iterable of (date, price, volume_or_None). Upserts all in one transaction."""
    data = [(card_id, str(d)[:10], float(p), None if v is None else int(v), source) for d, p, v in rows]
    if not data:
        raise ValueError("No prices to import.")
    for _, day, price, volume, _ in data:
        pd.Timestamp(day)
        if not math.isfinite(price) or price <= 0 or (volume is not None and not 0 <= volume < 2**63):
            raise ValueError("Prices must be finite and positive; volumes must be nonnegative SQLite integers.")
    with connect() as conn:
        removed = conn.execute("DELETE FROM prices WHERE card_id=? AND source='mock'", (card_id,)).rowcount if replace_mock else 0
        conn.executemany(
            """INSERT INTO prices (card_id, price_date, market_price, volume, catalyst_flag, source)
               VALUES (?,?,?,?,0,?)
               ON CONFLICT(card_id, price_date) DO UPDATE SET
                 market_price=excluded.market_price, volume=excluded.volume, source=excluded.source""",
            data)
    return (len(data), removed) if replace_mock else len(data)


def delete_price_rows(card_id, source=None):
    """Delete a card's price rows (optionally only those from one source). Returns rows deleted."""
    with connect() as conn:
        if source:
            cur = conn.execute("DELETE FROM prices WHERE card_id=? AND source=?", (card_id, source))
        else:
            cur = conn.execute("DELETE FROM prices WHERE card_id=?", (card_id,))
        return cur.rowcount


# ------------------------------------------------------------------ reads ----
def get_card(card_id):
    """Return a dict {card_id, name, rarity, set} or None if the card isn't in the database."""
    if not os.path.exists(DB_PATH):
        return None
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT card_id, name, rarity, set_name, api_id, variant FROM cards WHERE card_id = ?", (card_id,)
        ).fetchone()
    if row is None:
        return None
    return {"card_id": row[0], "name": row[1], "rarity": row[2], "set": row[3], "api_id": row[4], "variant": row[5]}


def list_cards() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(columns=["card_id", "name", "set_name", "rarity", "base_price", "api_id", "variant"])
    init_db()
    with connect() as conn:
        return pd.read_sql_query(
            "SELECT card_id, name, set_name, rarity, base_price, api_id, variant FROM cards ORDER BY name", conn)


def get_source_summary() -> pd.DataFrame:
    """One row per card: how many price points it has, from which source(s), and the latest one."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    init_db()
    with connect() as conn:
        return pd.read_sql_query(
            """SELECT c.card_id, c.name, c.api_id,
                      COUNT(p.price_date) AS points,
                      GROUP_CONCAT(DISTINCT p.source) AS sources,
                      MIN(p.price_date) AS first_date, MAX(p.price_date) AS last_date,
                      (SELECT market_price FROM prices WHERE card_id=c.card_id
                       ORDER BY price_date DESC LIMIT 1) AS last_price
               FROM cards c LEFT JOIN prices p ON p.card_id = c.card_id
               GROUP BY c.card_id ORDER BY c.name""", conn)


def get_price_history(card_id, days=None) -> pd.DataFrame:
    """Price history in the exact shape the model/app already expect:
    index 'Date'; columns 'Market Price ($)', 'Daily Trading Volume',
    'PSA 10 Multiplier', 'Catalyst Flag'. Empty frame if the card has no data.
    `days` keeps only the most recent N rows."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    init_db()
    with connect() as conn:
        df = pd.read_sql_query(
            """SELECT price_date AS Date, market_price AS "Market Price ($)",
                      volume AS "Daily Trading Volume", psa10_multiplier AS "PSA 10 Multiplier",
                      catalyst_flag AS "Catalyst Flag"
               FROM prices WHERE card_id = ? ORDER BY price_date""",
            conn, params=(card_id,))
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"])
    df["Catalyst Flag"] = df["Catalyst Flag"].astype(bool)
    df = df.set_index("Date")
    return df.iloc[-days:] if days else df


def get_card_sources(card_id):
    """(number_of_price_points, set_of_sources) for one card, e.g. (91, {'csv', 'tcgplayer'})."""
    if not os.path.exists(DB_PATH):
        return 0, set()
    with connect() as conn:
        n = conn.execute("SELECT COUNT(*) FROM prices WHERE card_id=?", (card_id,)).fetchone()[0]
        src = {r[0] for r in conn.execute("SELECT DISTINCT source FROM prices WHERE card_id=?", (card_id,))}
    return n, src
