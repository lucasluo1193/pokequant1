import numpy as np
import pandas as pd


class VolatilityForecaster:
    """Quant-style analytics for a card price series.

    The model is intentionally transparent: every component of the score is
    exposed so the app can explain why a signal was produced.
    """

    def __init__(self, price_series):
        self.prices = price_series.astype(float).dropna()
        if len(self.prices) < 2 or not np.isfinite(self.prices).all() or (self.prices <= 0).any():
            raise ValueError("Analysis requires at least two finite, positive prices.")
        if not isinstance(self.prices.index, pd.DatetimeIndex) or self.prices.index.hasnans:
            raise ValueError("Prices require valid calendar dates.")
        self.prices.index = self.prices.index.normalize()
        self.prices = self.prices.sort_index().groupby(level=0).last()
        if len(self.prices) < 2:
            raise ValueError("Analysis requires prices on at least two distinct dates.")
        self.prices = self.prices.asfreq("D").ffill()
        self.log_returns = np.log(self.prices / self.prices.shift(1)).dropna()

    @staticmethod
    def _safe(value, default=0.0):
        return float(value) if np.isfinite(value) else default

    def calculate_metrics(self):
        prices = self.prices
        returns = self.log_returns
        current = float(prices.iloc[-1])

        # Volatility: both simple rolling and EWMA.
        rolling_vol = returns.rolling(30).std() * np.sqrt(365)
        current_vol = self._safe(rolling_vol.iloc[-1], 0.20)
        ewma_daily = returns.ewm(span=30, adjust=False).std().iloc[-1]
        ewma_vol = self._safe(ewma_daily * np.sqrt(365), current_vol)

        # Multi-horizon momentum.
        def pct_change(days):
            if len(prices) <= days:
                return 0.0
            old = float(prices.iloc[-days - 1])
            return (current - old) / old * 100 if old else 0.0

        momentum_7 = pct_change(7)
        momentum_30 = pct_change(30)
        momentum_90 = pct_change(90)

        # RSI(14): useful for detecting unusually stretched short-term moves.
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = self._safe((100 - (100 / (1 + rs))).iloc[-1], 50.0)
        if loss.iloc[-1] == 0 and gain.iloc[-1] > 0:
            rsi = 100.0

        # Trend strength using two moving averages and their slope.
        ma10 = float(prices.tail(min(10, len(prices))).mean())
        ma30 = float(prices.tail(min(30, len(prices))).mean())
        ma90 = float(prices.tail(min(90, len(prices))).mean())
        trend_10_30 = (ma10 - ma30) / ma30 * 100 if ma30 else 0.0
        trend_30_90 = (ma30 - ma90) / ma90 * 100 if ma90 else 0.0

        # Drawdown from the recent 90-day high.
        recent_high = float(prices.tail(min(90, len(prices))).max())
        drawdown = (current - recent_high) / recent_high * 100 if recent_high else 0.0

        # Monte Carlo. Use log-return sampling and a fixed seed so the UI does
        # not jump around every Streamlit rerun.
        mean_return = self._safe(returns.mean(), 0.0)
        std_return = self._safe(returns.std(), 0.0001)
        rng = np.random.default_rng(42)
        simulated_returns = rng.normal(mean_return, std_return, (5000, 30))
        simulated_paths = current * np.exp(np.cumsum(simulated_returns, axis=1))
        final_prices = simulated_paths[:, -1]

        expected_roi = (final_prices.mean() - current) / current * 100
        median_roi = (np.median(final_prices) - current) / current * 100
        upside_prob = float(np.mean(final_prices > current * 1.05) * 100)
        downside_prob = float(np.mean(final_prices < current * 0.95) * 100)
        target_price = float(np.percentile(final_prices, 65))
        bear_price = float(np.percentile(final_prices, 20))
        bull_price = float(np.percentile(final_prices, 80))
        var_95 = current - float(np.percentile(final_prices, 5))

        # Sharpe-like measure, annualized. This is a descriptive statistic,
        # not a claim that card returns are normally distributed.
        annual_return = mean_return * 365
        annual_vol = std_return * np.sqrt(365)
        sharpe = annual_return / annual_vol if annual_vol else 0.0

        risk_score = int(np.clip(current_vol * 15, 1, 10))

        return {
            "current_volatility": round(current_vol * 100, 2),
            "ewma_volatility": round(ewma_vol * 100, 2),
            "var_95": round(var_95, 2),
            "expected_roi": round(expected_roi, 2),
            "median_roi": round(median_roi, 2),
            "upside_prob": round(upside_prob, 1),
            "downside_prob": round(downside_prob, 1),
            "target_price": round(target_price, 2),
            "bear_price": round(bear_price, 2),
            "bull_price": round(bull_price, 2),
            "risk_score": risk_score,
            "rsi": round(rsi, 1),
            "momentum_7": round(momentum_7, 2),
            "momentum_30": round(momentum_30, 2),
            "momentum_90": round(momentum_90, 2),
            "trend_10_30": round(trend_10_30, 2),
            "trend_30_90": round(trend_30_90, 2),
            "drawdown": round(drawdown, 2),
            "sharpe": round(sharpe, 2),
            "simulated_paths": simulated_paths,
        }

    def get_trading_signal(self, metrics):
        """Return a transparent -100 to +100 composite signal.

        Components:
          25 trend
          20 momentum
          20 expected return / target value
          15 RSI / mean-reversion context
          10 risk-adjusted performance
          10 probability edge
        """
        # Trend: both short-vs-medium and medium-vs-long confirmation.
        trend_score = np.clip(
            metrics["trend_10_30"] * 4 + metrics["trend_30_90"] * 2,
            -25, 25
        )

        # Momentum is intentionally capped so one huge price jump cannot take
        # over the entire model.
        momentum_score = np.clip(
            metrics["momentum_7"] * 0.8
            + metrics["momentum_30"] * 0.55
            + metrics["momentum_90"] * 0.25,
            -20, 20
        )

        roi_score = np.clip(metrics["expected_roi"] * 2.5, -20, 20)

        # RSI: mildly oversold is positive, mildly overbought is negative.
        # Extreme RSI is treated as a warning rather than an automatic trade.
        rsi = metrics["rsi"]
        if rsi < 30:
            rsi_score = 10
        elif rsi < 40:
            rsi_score = 5
        elif rsi > 75:
            rsi_score = -10
        elif rsi > 65:
            rsi_score = -5
        else:
            rsi_score = 0

        sharpe_score = np.clip(metrics["sharpe"] * 5, -10, 10)
        probability_score = np.clip(
            (metrics["upside_prob"] - metrics["downside_prob"]) * 0.35,
            -10, 10
        )

        raw = trend_score + momentum_score + roi_score + rsi_score + sharpe_score + probability_score
        total = int(round(np.clip(raw, -100, 100)))

        if total >= 25:
            signal = "BUY"
        elif total <= -25:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Confidence rises when the score is strong and several independent
        # signals agree. It falls when volatility is very high or indicators
        # disagree strongly.
        components = [trend_score / 25, momentum_score / 20, roi_score / 20,
                      rsi_score / 10, sharpe_score / 10, probability_score / 10]
        agreement = max(0.0, float(np.mean(np.sign(components))) * np.sign(total)) if total else 0.0
        confidence = 50 + abs(total) * 0.35 + agreement * 15 - max(metrics["current_volatility"] - 25, 0) * 0.15
        confidence = int(np.clip(confidence, 50, 95))

        current = float(self.prices.iloc[-1])
        target = metrics["target_price"]
        upside_to_target = (target - current) / current * 100 if current else 0.0

        reasons = [
            f"Trend confirmation: 10d/30d {metrics['trend_10_30']:+.2f}% and 30d/90d {metrics['trend_30_90']:+.2f}%",
            f"Momentum: 7d {metrics['momentum_7']:+.2f}% • 30d {metrics['momentum_30']:+.2f}% • 90d {metrics['momentum_90']:+.2f}%",
            f"30-day Monte Carlo: {metrics['upside_prob']:.1f}% chance of +5% and {metrics['downside_prob']:.1f}% chance of -5%",
            f"RSI(14): {metrics['rsi']:.1f} • risk score: {metrics['risk_score']}/10",
        ]

        return {
            "signal": signal,
            "score": total,
            "confidence": confidence,
            "reasons": reasons,
            "trend_pct": metrics["trend_10_30"],
            "momentum_pct": metrics["momentum_30"],
            "short_ma": round(float(self.prices.tail(min(10, len(self.prices))).mean()), 2),
            "long_ma": round(float(self.prices.tail(min(30, len(self.prices))).mean()), 2),
            "upside_to_target": round(upside_to_target, 2),
        }

    def forecast_jump_diffusion(self, horizon=30, n_paths=50):
        """Geometric Brownian Motion with a small jump component."""
        last_price = self.prices.iloc[-1]
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("Forecast horizon must be a positive integer.")
        dt = 1.0  # Estimates are daily log returns; each step is one calendar day.
        mu = self.log_returns.mean()
        sigma = self.log_returns.std()
        if not np.isfinite(sigma):
            sigma = 0.0

        rng = np.random.default_rng(7)
        lam = 0.05 / 365  # Expected jumps per calendar day.
        jump_mean = 0.0
        jump_std = 0.15

        paths = np.zeros((n_paths, horizon))
        for i in range(n_paths):
            path = [last_price]
            for _ in range(horizon):
                dz = rng.normal(0, np.sqrt(dt))
                jump = rng.poisson(lam * dt) * rng.normal(jump_mean, jump_std)
                price_next = path[-1] * np.exp(
                    mu * dt + sigma * dz + jump
                )
                path.append(price_next)
            paths[i, :] = path[1:]

        future_dates = pd.date_range(
            start=self.prices.index[-1] + pd.Timedelta(days=1),
            periods=horizon,
        )
        return future_dates, paths
