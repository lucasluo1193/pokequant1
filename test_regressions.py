import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import db
import price_sources as ps
from math_model import VolatilityForecaster
from card_images import find_card_image


class StorageTests(unittest.TestCase):
    def setUp(self):
        # Keep all temporary database writes and cleanup inside the workspace.
        self.temp = tempfile.TemporaryDirectory(dir=os.path.dirname(__file__))
        self.path_patch = patch.object(db, "DB_PATH", os.path.join(self.temp.name, "test.db"))
        self.path_patch.start()
        db.init_db()
        db.upsert_card("test", "Test", "Set", "Rare", 10)

    def tearDown(self):
        db._checked_paths.discard(db.DB_PATH)
        self.path_patch.stop()
        self.temp.cleanup()

    def test_existing_price_less_database_is_preserved(self):
        db.initialize_database()
        self.assertEqual(db.list_cards().card_id.tolist(), ["test"])

    def test_missing_database_is_seeded(self):
        with patch.object(db, "DB_PATH", os.path.join(self.temp.name, "new.db")):
            db.initialize_database()
            self.assertEqual(len(db.list_cards()), 12)

    def test_failed_csv_preserves_history(self):
        db.add_price("test", "2026-01-01", 10, source="mock")
        with self.assertRaises(ps.PriceSourceError):
            ps.import_csv("test", pd.DataFrame({"date": ["2026-01-02"], "price": [12], "volume": [np.inf]}))
        self.assertEqual(db.get_card_sources("test"), (1, {"mock"}))

    def test_database_failure_rolls_back_deletion(self):
        db.add_price("test", "2026-01-01", 10, source="mock")
        with db.connect() as conn:
            conn.execute("CREATE TRIGGER fail_import BEFORE INSERT ON prices BEGIN SELECT RAISE(ABORT, 'test failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            ps.import_csv("test", pd.DataFrame({"date": ["2026-01-02"], "price": [12]}))
        self.assertEqual(db.get_card_sources("test"), (1, {"mock"}))

    def test_daily_deduplication_and_successful_import(self):
        db.add_price("test", "2026-01-01", 10, source="mock")
        rows = pd.DataFrame({"date": ["2026-01-02 15:00", "2026-01-02 09:00"], "price": [12, 10]})
        self.assertEqual(len(ps.parse_csv(rows)), 1)
        self.assertEqual(ps.import_csv("test", rows), (1, 1))
        self.assertEqual(db.get_price_history("test").iloc[-1, 0], 12)

    @staticmethod
    def api_card():
        return {"id": "base1-4", "name": "Charizard", "set": {"name": "Base Set"}, "tcgplayer": {"prices": {"holofoil": {"market": 100}, "reverseHolofoil": {"market": 20}}}}

    def test_variants_have_separate_ids_and_refresh_prices(self):
        card = self.api_card()
        holo, _, _ = ps.save_card_with_price(card, "holofoil")
        reverse, _, _ = ps.save_card_with_price(card, "reverseHolofoil")
        self.assertNotEqual(holo, reverse)
        result = ps.refresh_tracked_cards(Mock(get_card=Mock(return_value=card)))
        self.assertTrue(result.Status.eq("ok").all())
        self.assertEqual(db.get_price_history(reverse).iloc[-1, 0], 20)
        del card["tcgplayer"]["prices"]["reverseHolofoil"]
        self.assertIsNone(ps.extract_price(card, "reverseHolofoil")[1])

    def test_legacy_refresh_skips_unconfirmed_variant(self):
        db.upsert_card("legacy", "Test", "Set", "Rare", 10, api_id="base1-4")
        client = Mock()
        result = ps.refresh_tracked_cards(client)
        self.assertIn("Legacy", result.iloc[0].Status)
        client.get_card.assert_not_called()

    def test_relinking_different_identity_preserves_real_history(self):
        db.add_price("test", "2026-01-01", 10, source="csv")
        with self.assertRaises(ValueError):
            db.link_real_card("test", "base1-4", "Other", "Other", "Rare", "holofoil")
        self.assertEqual(db.get_card("test")["name"], "Test")
        self.assertEqual(db.get_card_sources("test")[0], 1)

    def test_confirmed_relink_preserves_real_and_removes_mock(self):
        db.upsert_card("test", "Test", "Set", "Rare", 10, "base1-4", "holofoil")
        db.add_price("test", "2026-01-01", 10, source="csv")
        db.add_price("test", "2026-01-02", 12, source="mock")
        db.link_real_card("test", "base1-4", "Test", "Set", "Rare", "holofoil")
        self.assertEqual(db.get_card_sources("test"), (1, {"csv"}))

    def test_schema_migration_preserves_legacy_history(self):
        path = os.path.join(self.temp.name, "legacy.db")
        with closing(sqlite3.connect(path)) as conn:
            conn.executescript("CREATE TABLE cards(card_id TEXT PRIMARY KEY,name TEXT,set_name TEXT,rarity TEXT,base_price REAL,api_id TEXT); INSERT INTO cards VALUES('old','Old','Set','Rare',10,'base1-4');")
        with patch.object(db, "DB_PATH", path):
            db.init_db()
            self.assertIsNone(db.get_card("old")["variant"])

    def test_dashboard_renders_analysis(self):
        from streamlit.testing.v1 import AppTest
        db.add_prices_bulk("test", [(d.date().isoformat(), 10 + i * .1, None)
                                   for i, d in enumerate(pd.date_range("2026-01-01", periods=40))])
        with patch("card_images.find_card_image", return_value=None):
            app = AppTest.from_file("app.py").run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any(metric.label == "Recommendation" for metric in app.metric))


class AnalysisTests(unittest.TestCase):
    def test_invalid_csv_prices_are_rejected(self):
        for value in [np.inf, -np.inf, 0, -1]:
            with self.subTest(value=value), self.assertRaises(ps.PriceSourceError):
                ps.parse_csv(pd.DataFrame({"date": ["2026-01-01"], "price": [value]}))

    def test_invalid_volumes_are_rejected(self):
        for value in [np.inf, -1, 1.5, 2**63, "oops"]:
            with self.subTest(value=value), self.assertRaises(ps.PriceSourceError):
                ps.parse_csv(pd.DataFrame({"date": ["2026-01-01"], "price": [10], "volume": [value]}))

    def test_irregular_dates_are_daily_without_mutating_input(self):
        prices = pd.Series([10., 20., 30.], index=pd.date_range("2026-01-01", periods=3, freq="7D"))
        model = VolatilityForecaster(prices)
        self.assertEqual(len(model.prices), 15)
        self.assertEqual(model.calculate_metrics()["momentum_7"], 50)
        self.assertEqual(len(prices), 3)

    def test_flat_prices_have_neutral_confidence(self):
        model = VolatilityForecaster(pd.Series(10., index=pd.date_range("2026-01-01", periods=40)))
        signal = model.get_trading_signal(model.calculate_metrics())
        self.assertEqual((signal["score"], signal["confidence"]), (0, 50))

    def test_forecast_uses_daily_returns_and_simulates_first_date(self):
        prices = pd.Series(10 * np.exp(np.arange(40) * .01), index=pd.date_range("2026-01-01", periods=40))
        model = VolatilityForecaster(prices)
        dates, paths = model.forecast_jump_diffusion()
        self.assertEqual(paths.shape, (50, 30))
        self.assertEqual(dates[0], prices.index[-1] + pd.Timedelta(days=1))
        self.assertAlmostEqual(np.median(paths[:, 0]) / prices.iloc[-1], np.exp(.01))
        self.assertAlmostEqual(np.median(paths[:, -1]) / prices.iloc[-1], np.exp(.30))

    def test_invalid_model_prices_are_rejected(self):
        with self.assertRaises(ValueError):
            VolatilityForecaster(pd.Series([10, np.inf], index=pd.date_range("2026-01-01", periods=2)))


class ArtworkTests(unittest.TestCase):
    @patch("card_images.requests.get")
    def test_demo_artwork_uses_tcgdex_catalog_ids(self, get):
        expected = {
            "dragonite_holo": "https://assets.tcgdex.net/en/base/base3/4/high.webp",
            "gengar_holo": "https://assets.tcgdex.net/en/base/base3/5/high.webp",
            "mew_ex": "https://assets.tcgdex.net/en/sv/sv04.5/216/high.webp",
        }
        for card_id, url in expected.items():
            with self.subTest(card_id=card_id):
                self.assertEqual(find_card_image.__wrapped__("Demo", "Demo", card_id), url)
        get.assert_not_called()

    @patch("card_images.requests.get")
    def test_api_identity_overrides_demo_mapping(self, get):
        get.return_value.json.return_value = {"data": {"images": {"large": "exact.png"}}}
        self.assertEqual(find_card_image.__wrapped__("Test", "Set", "charizard_holo", "other-1"), "exact.png")
        self.assertIn("other-1", get.call_args.args[0])

    @patch("card_images.requests.get")
    def test_name_fallback_matches_set(self, get):
        get.return_value.json.return_value = [
            {"name": "Test", "set": {"name": "Wrong"}, "image": "wrong"},
            {"name": "Test", "set": {"name": "Right"}, "image": "right"},
        ]
        self.assertEqual(find_card_image.__wrapped__("Test", "Right"), "right/high.webp")
        self.assertIsNone(find_card_image.__wrapped__("Test", "Missing"))


if __name__ == "__main__":
    unittest.main()
