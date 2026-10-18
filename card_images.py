import re
import requests
import streamlit as st

# TCGdex provides a public card catalog and CDN. Unlike the old Pokémon TCG API,
# it does not require an API key. The CDN URL format is documented at tcgdex.dev.
TCGDEX_API = "https://api.tcgdex.net/v2/en/cards"
TCGDEX_ASSETS = "https://assets.tcgdex.net/en"

# The demo database intentionally uses friendly / made-up display names rather
# than exact TCG catalog names. These IDs connect those demo cards to real cards.
DEMO_CARD_IDS = {
    "charizard_holo": "base1-4",
    "blastoise_holo": "base1-2",
    "venusaur_holo": "base1-15",
    "pikachu_promo": "basep-1",
    "mewtwo_holo": "base1-10",
    "gengar_holo": "base3-5",
    "dragonite_holo": "base3-4",
    "lugia_holo": "neo1-9",
    "umbreon_alt_art": "swsh7-215",
    "rayquaza_vmax": "swsh7-111",
    "mew_ex": "sv04.5-216",
    "gyarados_holo": "base1-6",
}


def _tcgdex_image(card_id):
    """Return a real card image from the TCGdex CDN for a TCGdex card id."""
    if not card_id or "-" not in card_id:
        return None
    set_id, local_id = card_id.split("-", 1)

    # TCGdex separates the series from the set in its asset path. For the
    # common sets below, this is enough to build a stable image URL.
    series = {
        "base1": "base",
        "basep": "base",
        "base3": "base",
        "neo1": "neo",
        "swsh7": "swsh",
        "sv04.5": "sv",
    }.get(set_id)
    if not series:
        return None

    return f"{TCGDEX_ASSETS}/{series}/{set_id}/{local_id}/high.webp"


@st.cache_data(ttl=86400, show_spinner=False)
def find_card_image(name, set_name, card_id=None, api_id=None):
    """Find artwork for seeded cards or cards added through the database UI."""
    # API-linked cards use their original catalog identity, even when a demo
    # database row has been repurposed. Never substitute a name-only match.
    if api_id:
        try:
            response = requests.get(f"https://api.pokemontcg.io/v2/cards/{api_id}", timeout=10)
            response.raise_for_status()
            card = response.json().get("data", {})
            return (card.get("images") or {}).get("large") or (card.get("images") or {}).get("small")
        except (requests.RequestException, ValueError, TypeError, AttributeError):
            return None
    # 1. Exact mapping for seeded/demo cards.
    if card_id in DEMO_CARD_IDS:
        image = _tcgdex_image(DEMO_CARD_IDS[card_id])
        if image:
            return image

    if card_id and "-" in str(card_id):
        try:
            response = requests.get(f"{TCGDEX_API}/{card_id}", timeout=8)
            response.raise_for_status()
            exact_card = response.json()
            if isinstance(exact_card, dict) and exact_card.get("image"):
                return exact_card["image"].rstrip("/") + "/high.webp"
        except (requests.RequestException, ValueError, TypeError):
            pass

    # 2. TCGdex REST filters use `name=...`; `q=...` is not a supported
    # filter and can silently return unrelated cards or no useful results.
    # Database display names sometimes include labels such as "Holo" or
    # "Alt Art", so try both the full name and a cleaned Pokémon/card name.
    raw_name = (name or "").strip()
    cleaned_name = re.sub(
        r"\s+(holo|holo rare|reverse holo|alt art|alternate art|full art|\(.*?\))$",
        "", raw_name, flags=re.IGNORECASE,
    ).strip()
    search_names = []
    for candidate in (raw_name, cleaned_name):
        if candidate and candidate.casefold() not in [n.casefold() for n in search_names]:
            search_names.append(candidate)

    for search_name in search_names:
        try:
            response = requests.get(
                TCGDEX_API,
                params={"name": search_name, "pagination:itemsPerPage": 100},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                data = data.get("cards", data.get("data", []))
            if not isinstance(data, list) or not data:
                continue

            # List results are CardBrief records; set metadata may be absent.
            # Prefer an exact card-name match, then use the first image result.
            candidates = [c for c in data if isinstance(c, dict) and c.get("image")]
            if not candidates:
                continue
            for candidate in candidates:
                if str(candidate.get("name", "")).casefold() not in {raw_name.casefold(), cleaned_name.casefold()}:
                    continue
                detail = candidate
                if not isinstance(detail.get("set"), dict):
                    if not candidate.get("id"):
                        continue
                    detail_response = requests.get(f"{TCGDEX_API}/{candidate['id']}", timeout=8)
                    detail_response.raise_for_status()
                    detail = detail_response.json()
                if not isinstance(detail, dict):
                    continue
                if str((detail.get("set") or {}).get("name", "")).casefold() != (set_name or "").casefold():
                    continue
                image_base = detail.get("image")
                if image_base:
                    return image_base.rstrip("/") + "/high.webp"
        except (requests.RequestException, ValueError, TypeError):
            continue

    # 3. If a database row's ID is already a TCGdex ID (for example base1-4),
    # use it directly as a final fallback.
    if card_id and "-" in str(card_id):
        try:
            response = requests.get(f"{TCGDEX_API}/{card_id}", timeout=8)
            if response.ok:
                card = response.json()
                image_base = card.get("image") if isinstance(card, dict) else None
                if image_base:
                    return image_base.rstrip("/") + "/high.webp"
        except (requests.RequestException, ValueError, TypeError):
            pass

    return None
