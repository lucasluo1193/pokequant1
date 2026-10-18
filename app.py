import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import db
import data_generator
import price_ui
from math_model import VolatilityForecaster
from card_images import find_card_image

st.set_page_config(page_title="PokeQuant: Quant Card Trader", layout="wide")

MIN_HISTORY = 35


@st.cache_resource
def init_database():
    # First run: build the starter database from seed_cards.csv.
    db.initialize_database()


init_database()

st.title("PokeQuant: Quant Card Trader")
st.markdown(
    "Search your Pokémon card database, view real market prices, and get a "
    "quant-style **BUY / HOLD / SELL** signal based on price momentum, trend, "
    "volatility, and simulated expected return."
)

tab_database, tab_prices = st.tabs(["Card Database", "Real Prices"])

# ------------------------------------------------------------------ database
with tab_database:
    st.subheader("Card Database")

    all_cards = db.list_cards()

    if all_cards.empty:
        st.warning("There are no cards in the database yet. Add one from the Real Prices tab.")
    else:
        search = st.text_input(
            "Search the database",
            placeholder="Search by card name, set, rarity, or card ID..."
        ).strip().lower()

        filtered = all_cards.copy()
        if search:
            mask = (
                filtered["name"].fillna("").str.lower().str.contains(search, regex=False)
                | filtered["set_name"].fillna("").str.lower().str.contains(search, regex=False)
                | filtered["rarity"].fillna("").str.lower().str.contains(search, regex=False)
                | filtered["card_id"].fillna("").str.lower().str.contains(search, regex=False)
            )
            filtered = filtered[mask]

        st.caption(f"{len(filtered)} card(s) found.")

        if filtered.empty:
            st.info("No cards matched your search.")
        else:
            display = filtered[["name", "set_name", "rarity", "variant", "base_price", "card_id"]].copy()
            display.columns = ["Card", "Set", "Rarity", "Variant", "Starting Price ($)", "Card ID"]
            st.dataframe(display, hide_index=True, use_container_width=True)

            selected_id = st.selectbox(
                "Open a card",
                filtered["card_id"].tolist(),
                format_func=lambda cid: (
                    f"{filtered.loc[filtered['card_id'] == cid, 'name'].iloc[0]} "
                    f"— {filtered.loc[filtered['card_id'] == cid, 'set_name'].iloc[0]}"
                    f" ({filtered.loc[filtered['card_id'] == cid, 'variant'].iloc[0] or 'unspecified variant'})"
                )
            )

            card = db.get_card(selected_id)

            if card:
                st.markdown("---")
                st.subheader(f"{card['name']}")

                # Pull the card artwork automatically after selection. No camera
                # or upload is needed anymore.
                image_url = find_card_image(card["name"], card["set"], selected_id, card.get("api_id"))
                info_col, image_col = st.columns([2.2, 1])
                with info_col:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Card", card["name"])
                    c2.metric("Rarity", card["rarity"])
                    c3.metric("Set", card["set"])
                with image_col:
                    if image_url:
                        st.image(image_url, caption=card["name"], use_container_width=True)
                    else:
                        st.info("Card artwork unavailable")

                df_history = db.get_price_history(selected_id, days=180)

                if df_history.empty:
                    st.warning("No price history is available for this card yet.")
                else:
                    prices = df_history["Market Price ($)"]
                    n_points, sources = db.get_card_sources(selected_id)

                    if "mock" in sources:
                        st.caption(
                            f"Data: made-up ({n_points} price points)"
                            + (" + real" if sources - {"mock"} else "")
                        )
                    else:
                        st.caption(
                            f"Data: real prices ({n_points} price points; "
                            f"source: {', '.join(sorted(sources))})"
                        )

                    # ------------------------------------------------ signal
                    if len(prices) < MIN_HISTORY:
                        st.metric("Latest price", f"${prices.iloc[-1]:,.2f}")
                        st.info(
                            f"This card has {len(prices)} price point(s). "
                            f"The quant signal needs at least {MIN_HISTORY} points."
                        )
                        if len(prices) >= 2:
                            st.line_chart(prices)
                    else:
                        if prices.index.to_series().diff().dropna().ne(pd.Timedelta(days=1)).any():
                            st.caption("Missing calendar days are forward-filled for analysis; no observations are added to storage.")
                        forecaster = VolatilityForecaster(prices)
                        metrics = forecaster.calculate_metrics()
                        signal = forecaster.get_trading_signal(metrics)

                        st.markdown("### Quant Trading Signal")

                        signal_col, score_col, confidence_col, target_col = st.columns(4)
                        signal_col.metric("Recommendation", signal["signal"])
                        score_col.metric("Quant Score", f"{signal['score']:+d} / 100")
                        confidence_col.metric("Model Confidence", f"{signal['confidence']}%")
                        target_col.metric("30-Day Target", f"${metrics['target_price']:,.2f}", f"{signal['upside_to_target']:+.2f}%")

                        st.progress((signal["score"] + 100) / 200)

                        with st.expander("Open the model breakdown", expanded=True):
                            b1, b2, b3 = st.columns(3)
                            b1.metric("7-Day Momentum", f"{metrics['momentum_7']:+.2f}%")
                            b1.metric("30-Day Momentum", f"{metrics['momentum_30']:+.2f}%")
                            b2.metric("RSI (14)", f"{metrics['rsi']:.1f}")
                            b2.metric("Trend 10d / 30d", f"{metrics['trend_10_30']:+.2f}%")
                            b3.metric("Trend 30d / 90d", f"{metrics['trend_30_90']:+.2f}%")
                            b3.metric("Drawdown", f"{metrics['drawdown']:+.2f}%")

                            st.write("**Signal drivers**")
                            for reason in signal["reasons"]:
                                st.write(f"• {reason}")

                            st.write("**Monte Carlo scenario range**")
                            sc1, sc2, sc3, sc4 = st.columns(4)
                            sc1.metric("Bear Case (20th)", f"${metrics['bear_price']:,.2f}")
                            sc2.metric("Median", f"{metrics['median_roi']:+.2f}%")
                            sc3.metric("Target (65th)", f"${metrics['target_price']:,.2f}")
                            sc4.metric("Bull Case (80th)", f"${metrics['bull_price']:,.2f}")

                        st.caption(
                            "Quant score is an educational, rules-based model using historical card prices. "
                            "It is not a guarantee of future prices or financial advice."
                        )

                        st.markdown("---")
                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric("Current Volatility", f"{metrics['current_volatility']}%")
                        m2.metric("EWMA Volatility", f"{metrics['ewma_volatility']}%")
                        m3.metric("95% VaR (30d)", f"${metrics['var_95']}")
                        m4.metric("Expected ROI", f"{metrics['expected_roi']:+.2f}%")
                        m5.metric("Risk Score", f"{metrics['risk_score']} / 10")

                        # ------------------------------------------ price chart
                        st.markdown("### Price Forecast")
                        ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
                        horizon = ctrl1.select_slider(
                            "Forecast horizon (days)", options=[7, 14, 30, 60, 90], value=30
                        )
                        n_shown = ctrl2.slider("Simulation paths shown", 0, 50, 10)
                        show_bands = ctrl3.toggle("Confidence bands", value=True)

                        # Many paths give smooth percentile bands; only a few are drawn.
                        future_dates, mc_paths = forecaster.forecast_jump_diffusion(horizon=horizon, n_paths=500)
                        last_date = df_history.index[-1]
                        last_price = prices.iloc[-1]
                        band_x = [last_date, *future_dates]

                        def from_today(values):
                            return [last_price, *values]

                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=df_history.index,
                                y=prices,
                                mode="lines",
                                name="Historical Price",
                                line=dict(width=2, color="#6366f1"),
                                hovertemplate="Historical Price<br>%{x|%b %d, %Y}<br>$%{y:,.2f}<extra></extra>",
                            )
                        )

                        if show_bands:
                            p10, p25, p75, p90 = np.percentile(mc_paths, [10, 25, 75, 90], axis=0)
                            for low, high, label, alpha in (
                                (p10, p90, "80% range (10th–90th)", 0.12),
                                (p25, p75, "50% range (25th–75th)", 0.22),
                            ):
                                fig.add_trace(
                                    go.Scatter(
                                        x=band_x, y=from_today(low), mode="lines",
                                        line=dict(width=0), showlegend=False, hoverinfo="skip",
                                    )
                                )
                                fig.add_trace(
                                    go.Scatter(
                                        x=band_x, y=from_today(high), mode="lines",
                                        line=dict(width=0), fill="tonexty",
                                        fillcolor=f"rgba(56, 189, 248, {alpha})",
                                        name=label, hoverinfo="skip",
                                    )
                                )

                        # Faint individual simulations; hovering one names it.
                        for i in range(min(n_shown, mc_paths.shape[0])):
                            path_return = (mc_paths[i] / last_price - 1) * 100
                            fig.add_trace(
                                go.Scatter(
                                    x=band_x,
                                    y=from_today(mc_paths[i]),
                                    mode="lines",
                                    name="Monte Carlo Simulations",
                                    legendgroup="simulations",
                                    showlegend=(i == 0),
                                    line=dict(width=1, color="rgba(150, 150, 150, 0.25)"),
                                    customdata=[0.0, *path_return],
                                    hovertemplate=(
                                        f"Simulation {i + 1}<br>%{{x|%b %d, %Y}}<br>"
                                        "$%{y:,.2f} (%{customdata:+.2f}%)<extra></extra>"
                                    ),
                                )
                            )

                        # Main line: average across all simulated paths.
                        expected_path = mc_paths.mean(axis=0)
                        expected_return = (expected_path / last_price - 1) * 100
                        trend_color = "#22c55e" if expected_return[-1] >= 0 else "#ef4444"
                        fig.add_trace(
                            go.Scatter(
                                x=band_x,
                                y=from_today(expected_path),
                                mode="lines",
                                name=f"Expected Path ({expected_return[-1]:+.2f}%)",
                                line=dict(width=3, color=trend_color),
                                customdata=[0.0, *expected_return],
                                hovertemplate=(
                                    "Expected Path<br>%{x|%b %d, %Y}<br>"
                                    "$%{y:,.2f} (%{customdata:+.2f}%)<extra></extra>"
                                ),
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=[future_dates[-1]],
                                y=[expected_path[-1]],
                                mode="markers+text",
                                marker=dict(size=10, color=trend_color, line=dict(width=2, color="white")),
                                text=[f"${expected_path[-1]:,.2f}"],
                                textposition="top center",
                                textfont=dict(color=trend_color, size=13),
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )

                        fig.add_shape(
                            type="line", x0=last_date, x1=last_date, y0=0, y1=1, yref="paper",
                            line=dict(color="rgba(150, 150, 150, 0.6)", width=1, dash="dot"),
                        )
                        fig.add_annotation(
                            x=last_date, y=0, yref="paper", text="Today", showarrow=False,
                            yanchor="bottom", xanchor="right", font=dict(size=11, color="gray"),
                        )

                        fig.update_layout(
                            title=f"180-Day History & {horizon}-Day Forecast — {card['name']}",
                            yaxis_title="Price (USD)",
                            yaxis_tickprefix="$",
                            height=520,
                            hovermode="closest",
                            margin=dict(t=90),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                            xaxis=dict(
                                rangeselector=dict(
                                    buttons=[
                                        dict(count=1, label="1M", step="month", stepmode="backward"),
                                        dict(count=3, label="3M", step="month", stepmode="backward"),
                                        dict(step="all", label="All"),
                                    ],
                                    x=1,
                                    xanchor="right",
                                    y=1.12,
                                ),
                                rangeslider=dict(visible=True, thickness=0.06),
                            ),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        end_col1, end_col2, end_col3 = st.columns(3)
                        end_col1.metric(f"Expected price in {horizon}d", f"${expected_path[-1]:,.2f}", f"{expected_return[-1]:+.2f}%")
                        end_col2.metric("Chance of gain", f"{(mc_paths[:, -1] > last_price).mean() * 100:.0f}%")
                        end_col3.metric("80% range", f"\\${np.percentile(mc_paths[:, -1], 10):,.2f} – \\${np.percentile(mc_paths[:, -1], 90):,.2f}")

# --------------------------------------------------------------- real prices
with tab_prices:
    price_ui.render()
