"""
price_sources.py - load REAL card prices into the database.

Source: the Pokemon TCG API (pokemontcg.io). Every card it returns includes TCGplayer
prices (low / mid / high / market) per print variant (holofoil, normal, ...). It works
without an API key (lower rate limit); a free key from https://dev.pokemontcg.io raises it.

Important: the API only returns TODAY's price, not history. So "recording" a price adds
one data point for today; history builds up each day you refresh. For past prices, use the
CSV importer (import_csv) with data from a site that offers exports.
"""
import os
import re
import time
import math
from datetime import date

import pandas as pd
import numpy as np
import requests

import db

API_URL = "https://api.pokemontcg.io/v2"
# Which print variant to read when a card has several, best guess first.
VARIANT_PREFERENCE = [
    "holofoil", "1stEditionHolofoil", "unlimitedHolofoil", "normal", "1stEditionNormal",
    "unlimitedNormal", "reverseHolofoil",
]


class PriceSourceError(Exception):
    """Any problem fetching/reading prices, with a message that is safe to show the user."""


# ------------------------------------------------------------------- client ----
class PokemonTCGClient:
    def __init__(self, api_key=None, timeout=30, retries=2):
        self.api_key = api_key or os.environ.get("POKEMONTCG_API_KEY") or None
        self.timeout = timeout
        self.retries = retries

    def _get(self, path, params=None):
        headers = {"X-Api-Key": self.api_key} if self.api_key else {}
        last = None
        for attempt in range(self.retries + 1):
            try:
                r = requests.get(f"{API_URL}/{path}", params=params, headers=headers, timeout=self.timeout)
            except requests.exceptions.Timeout:
                last = "pokemontcg.io took too long to respond (it is sometimes slow). Try again in a moment."
            except requests.exceptions.ConnectionError:
                last = "Couldn't reach pokemontcg.io - check your internet connection."
            except requests.exceptions.RequestException as e:
                last = f"Request failed: {e}"
            else:
                if r.status_code == 200:
                    try:
                        return r.json()
                    except ValueError:
                        last = "pokemontcg.io returned something that isn't valid JSON."
                elif r.status_code == 404:
                    raise PriceSourceError("Card not found on pokemontcg.io.")
                elif r.status_code == 429:
                    last = "Rate limit hit. Wait a minute, or add a free API key to raise the limit."
                elif r.status_code in (401, 403):
                    raise PriceSourceError("The API key was rejected - check it or clear the field.")
                else:
                    last = f"pokemontcg.io returned an error (HTTP {r.status_code})."
            if attempt < self.retries:
                time.sleep(1.5 * (attempt + 1))
        raise PriceSourceError(last or "Unknown error contacting pokemontcg.io.")

    def search(self, text, limit=24):
        """Search by card name ('charizard', 'charizard ex') or by exact id ('base1-4')."""
        text = text.strip()
        if not text:
            return []
        if re.fullmatch(r"[a-z0-9]+-[a-z0-9]+", text.lower()) and any(ch.isdigit() for ch in text):
            return [self.get_card(text.lower())]
        words = re.sub(r'[^A-Za-z0-9 \'\-é]', " ", text).split()
        q = f"name:{words[0]}*" if len(words) == 1 else 'name:"' + " ".join(words) + '"'
        data = self._get("cards", {
            "q": q, "pageSize": limit, "orderBy": "-set.releaseDate",
            "select": "id,name,number,rarity,set,images,tcgplayer",
        })
        return data.get("data", [])

    def get_card(self, api_id):
        data = self._get(f"cards/{api_id}", {"select": "id,name,number,rarity,set,images,tcgplayer"})
        return data["data"]


# ------------------------------------------------------------------ parsing ----
def available_variants(card):
    """Variants that actually have a price, in a sensible order."""
    prices = ((card.get("tcgplayer") or {}).get("prices")) or {}
    have = [v for v, p in prices.items() if isinstance(p, dict) and _valid_price(p) is not None]
    return sorted(have, key=lambda v: VARIANT_PREFERENCE.index(v) if v in VARIANT_PREFERENCE else 99)


def _valid_price(prices):
    for field in ("market", "mid"):
        try:
            value = float(prices.get(field))
            if math.isfinite(value) and value > 0:
                return value
        except (TypeError, ValueError):
            pass
    return None


def extract_price(card, variant=None):
    """Return (variant, price) from a pokemontcg.io card dict. Uses TCGplayer's *market*
    price, falling back to the mid price. Returns (None, None) if there is no price."""
    prices = ((card.get("tcgplayer") or {}).get("prices")) or {}
    variants = available_variants(card)
    if not variants:
        return None, None
    if variant is not None and variant not in variants:
        return None, None
    v = variant if variant in variants else variants[0]
    p = prices[v]
    return v, _valid_price(p)


def make_card_id(card):
    """Stable slug for the database, e.g. 'charizard_base1_4'."""
    name = re.sub(r"[^a-z0-9]+", "_", card["name"].lower()).strip("_")
    ident = re.sub(r"[^a-z0-9]+", "_", card["id"].lower()).strip("_")
    return f"{name}_{ident}"


def card_details(card):
    s = card.get("set") or {}
    return {
        "name": card["name"], "set_name": s.get("name", "Unknown set"),
        "rarity": card.get("rarity") or "Unknown", "api_id": card["id"],
    }


# ----------------------------------------------------------------- database ----
def save_card_with_price(card, variant=None, card_id=None, link_existing=False):
    """Store a card from the API and record today's TCGplayer price for it.
    - card_id=None            -> create a new card (slug made from name + id)
    - card_id + link_existing -> turn an existing (mock) card into this real card; its MOCK
                                 price rows are deleted so fake and real data never mix.
    Returns (card_id, variant, price)."""
    variant, price = extract_price(card, variant)
    if price is None:
        raise PriceSourceError("That card has no TCGplayer price on pokemontcg.io.")
    d = card_details(card)
    if link_existing and card_id:
        try:
            db.link_real_card(card_id, d["api_id"], d["name"], d["set_name"], d["rarity"], variant)
        except ValueError as e:
            raise PriceSourceError(str(e)) from e
    else:
        card_id = card_id or f"{make_card_id(card)}__{variant}"
        existing = db.get_card(card_id)
        if existing and (existing["api_id"] != d["api_id"] or existing["variant"] != variant):
            raise PriceSourceError("That card ID belongs to a different or unconfirmed variant. Save with a new ID.")
        db.upsert_card(card_id, d["name"], d["set_name"], d["rarity"], price, api_id=d["api_id"], variant=variant)
    db.add_price(card_id, date.today().isoformat(), price, source="tcgplayer")
    return card_id, variant, price


def refresh_tracked_cards(client, progress=None):
    """Record today's price for every card that has a pokemontcg.io id. `progress(i, n, name)`
    is called after each card. Returns a DataFrame with one row per card (ok / error)."""
    tracked = db.list_cards()
    tracked = tracked[tracked["api_id"].notna() & (tracked["api_id"] != "")]
    out = []
    for i, row in enumerate(tracked.itertuples(index=False), 1):
        try:
            if not row.variant:
                raise PriceSourceError("Legacy card has no confirmed print variant. Save it as a new card with the correct variant to begin a separate history.")
            card = client.get_card(row.api_id)
            variant, price = extract_price(card, row.variant)
            if price is None:
                raise PriceSourceError("no TCGplayer price available")
            db.add_price(row.card_id, date.today().isoformat(), price, source="tcgplayer")
            out.append({"Card": row.name, "Price": price, "Variant": variant, "Status": "ok"})
        except PriceSourceError as e:
            out.append({"Card": row.name, "Price": None, "Variant": None, "Status": f"failed: {e}"})
        if progress:
            progress(i, len(tracked), row.name)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------- CSV ----
DATE_COLS = ["date", "price_date", "timestamp", "day", "time"]
PRICE_COLS = ["price", "market_price", "market price ($)", "market price", "market", "close", "value", "usd"]
VOLUME_COLS = ["volume", "daily trading volume", "sales", "qty"]


def _find(df, names):
    lower = {str(c).lower().strip(): c for c in df.columns}
    for n in names:
        if n in lower:
            return lower[n]
    return None


def parse_csv(df):
    """Detect the date / price / volume columns of an uploaded table. Returns a clean
    DataFrame with columns date, price, volume (volume may be NaN) or raises PriceSourceError."""
    dcol, pcol, vcol = _find(df, DATE_COLS), _find(df, PRICE_COLS), _find(df, VOLUME_COLS)
    if dcol is None or pcol is None:
        raise PriceSourceError(
            "Couldn't find the columns. The file needs a date column (e.g. 'date') and a price "
            f"column (e.g. 'price'). Columns found: {', '.join(map(str, df.columns))}")
    out = pd.DataFrame({
        "date": pd.to_datetime(df[dcol], errors="coerce", format="mixed", utc=True).dt.tz_convert(None),
        "price": pd.to_numeric(df[pcol].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce"),
        "volume": pd.to_numeric(df[vcol], errors="coerce") if vcol else float("nan"),
    }).dropna(subset=["date", "price"])
    out = out[np.isfinite(out["price"]) & (out["price"] > 0)].copy()
    volume = out["volume"]
    if (volume.notna() & (~np.isfinite(volume) | (volume < 0) | (volume >= 2**63) | (volume % 1 != 0))).any():
        raise PriceSourceError("Volume must be a finite, nonnegative whole number below 2^63, or blank.")
    if vcol:
        supplied = df.loc[out.index, vcol]
        if (supplied.notna() & supplied.astype(str).str.strip().ne("") & volume.isna()).any():
            raise PriceSourceError("Volume contains an unreadable number.")
    out = out.sort_values("date", kind="stable")
    out["date"] = out["date"].dt.normalize()
    out = out.drop_duplicates("date", keep="last")
    if out.empty:
        raise PriceSourceError("No valid rows found (need a readable date and a price above 0).")
    return out.reset_index(drop=True)


def import_csv(card_id, df, replace_mock=True):
    """Load parsed CSV rows for a card. If replace_mock, the card's made-up rows are deleted first."""
    clean = parse_csv(df)
    rows = [(r.date.date().isoformat(), r.price, None if pd.isna(r.volume) else r.volume)
            for r in clean.itertuples(index=False)]
    result = db.add_prices_bulk(card_id, rows, source="csv", replace_mock=replace_mock)
    return result if replace_mock else (result, 0)
