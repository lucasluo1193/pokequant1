"""
price_ui.py - the "Load Real Prices" tab of the Streamlit app.
All the real work happens in price_sources.py / db.py; this file is only the screen.
"""
import os

import pandas as pd
import streamlit as st

import db
import price_sources as ps


def _forget_cached_scanner():
    # The scanner caches card names/rarities when it starts; reload it after the database changes.
    st.cache_resource.clear()


def render():
    st.subheader("Load real prices")
    st.info(
        "Prices come from the **Pokémon TCG API (pokemontcg.io)**, which includes TCGplayer's "
        "market price for each card. It only gives **today's** price, so each time you record "
        "(or press *Update all*), one new data point is added. The volatility model needs about "
        "35 days of data - build it up daily, or import past prices from a CSV below."
    )
    key = st.text_input(
        "API key (optional)", type="password", value=os.environ.get("POKEMONTCG_API_KEY", ""),
        help="Works without one. A free key from dev.pokemontcg.io raises the rate limit.")
    client = ps.PokemonTCGClient(api_key=key.strip() or None)

    # ------------------------------------------------ 1. search & add a card
    st.markdown("### 1. Find a card and record its TCGplayer price")
    q = st.text_input("Card name or pokemontcg.io id", placeholder="charizard, charizard ex, base1-4 ...")
    if st.button("Search") and q.strip():
        try:
            with st.spinner("Searching pokemontcg.io..."):
                st.session_state["pq_results"] = client.search(q)
        except ps.PriceSourceError as e:
            st.session_state["pq_results"] = []
            st.error(str(e))

    results = [c for c in st.session_state.get("pq_results", []) if ps.available_variants(c)]
    raw_count = len(st.session_state.get("pq_results", []))
    if raw_count and not results:
        st.warning("Found cards, but none of them have a TCGplayer price.")
    if results:
        st.caption(f"{raw_count} cards found, {len(results)} with TCGplayer prices.")
        summary = pd.DataFrame([{
            "Card": c["name"], "Set": (c.get("set") or {}).get("name", ""), "No.": c.get("number", ""),
            "Rarity": c.get("rarity", ""), "Market ($)": ps.extract_price(c)[1], "Id": c["id"],
        } for c in results])
        st.dataframe(summary, hide_index=True)

        idx = st.selectbox("Pick a card", range(len(results)),
                           format_func=lambda i: f"{results[i]['name']} - {(results[i].get('set') or {}).get('name', '')} "
                                                 f"#{results[i].get('number', '')}")
        card = results[idx]
        c1, c2 = st.columns([1, 3])
        img = (card.get("images") or {}).get("small")
        if img:
            c1.image(img, width=150)
        with c2:
            variant = st.selectbox("Print variant", ps.available_variants(card))
            _, price = ps.extract_price(card, variant)
            st.metric("TCGplayer market price", f"${price:,.2f}")
            mode = st.radio("Save as", ["A new card", "Replace an existing card in the database"])
            target = None
            if mode != "A new card":
                existing = db.list_cards()
                target = st.selectbox("Card to replace", list(existing["card_id"]),
                                      format_func=lambda cid: existing.set_index("card_id").loc[cid, "name"])
                st.warning("This updates card details and deletes demo history. Cards with real history can only keep the same confirmed API ID and print variant; otherwise save a new card.")
            if st.button("Save card and record today's price", type="primary"):
                try:
                    cid, v, p = ps.save_card_with_price(card, variant, card_id=target, link_existing=target is not None)
                    _forget_cached_scanner()
                    st.success(f"Saved {card['name']} ({v}) at ${p:,.2f}. Card id: `{cid}`")
                except ps.PriceSourceError as e:
                    st.error(str(e))

    # ------------------------------------------------ 2. refresh everything
    st.markdown("### 2. Update all tracked cards")
    tracked = db.list_cards()
    tracked = tracked[tracked["api_id"].notna() & (tracked["api_id"] != "")]
    st.caption(f"{len(tracked)} card(s) are linked to real prices. Run this once a day to build history.")
    if not tracked.empty and tracked["variant"].isna().any():
        st.info("Some older cards have no recorded print variant. Their history is preserved, but refresh is skipped. Save a new card with the correct variant to start tracking it separately.")
    if st.button("Update all prices", disabled=tracked.empty):
        bar = st.progress(0.0)
        out = ps.refresh_tracked_cards(
            client, progress=lambda i, n, name: bar.progress(i / max(n, 1), text=f"{name} ({i}/{n})"))
        st.dataframe(out, hide_index=True)

    # ------------------------------------------------ 3. CSV import
    st.markdown("### 3. Import past prices from a CSV")
    st.caption("Any CSV with a date column and a price column (extra columns are ignored; "
               "a volume column is used if present).")
    cards = db.list_cards()
    if cards.empty:
        st.caption("Add a card first.")
    else:
        cid = st.selectbox("Import into card", list(cards["card_id"]), key="csv_card",
                           format_func=lambda c: cards.set_index("card_id").loc[c, "name"])
        up = st.file_uploader("CSV file", type=["csv"], key="csv_file")
        replace = st.checkbox("Delete this card's made-up price history first", value=True)
        if up is not None:
            try:
                parsed = ps.parse_csv(pd.read_csv(up))
                st.caption(f"Found {len(parsed)} valid rows from {parsed['date'].min().date()} "
                           f"to {parsed['date'].max().date()}.")
                st.dataframe(parsed.head(5), hide_index=True)
                if st.button("Import these prices"):
                    n, removed = ps.import_csv(cid, parsed.rename(columns={"date": "date", "price": "price"}),
                                               replace_mock=replace)
                    st.success(f"Imported {n} prices" + (f" and removed {removed} made-up ones." if removed else "."))
            except ps.PriceSourceError as e:
                st.error(str(e))
            except Exception as e:  # unreadable file
                st.error(f"Couldn't read that file: {e}")

    # ------------------------------------------------ 4. status
    st.markdown("### Data status")
    status = db.get_source_summary()
    if not status.empty:
        status["Data"] = status["sources"].fillna("none").apply(
            lambda s: "made-up" if s == "mock" else ("none" if s == "none" else "real" + (" + made-up" if "mock" in s else "")))
        st.dataframe(status.rename(columns={
            "name": "Card", "points": "Price points", "first_date": "From", "last_date": "To",
            "last_price": "Last price ($)", "api_id": "API id"})[
            ["Card", "Data", "Price points", "From", "To", "Last price ($)", "API id"]], hide_index=True)
