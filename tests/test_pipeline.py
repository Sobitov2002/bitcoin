"""
tests/test_pipeline.py — Unit Tests for Bitcoin AI Dashboard.

Tests:
1. Feature engineering and data integrity.
2. Model bundle structure, pipeline loading, and prediction outputs.
3. Database manager, local fallback, and accuracy calculation.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import app
import db_manager


class TestPipeline(unittest.TestCase):

    def test_feature_engineering(self):
        """Verify that build_features creates all required features without error."""
        timestamps = pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC")
        prices = 50000.0 + np.cumsum(np.random.randn(100) * 100)
        volumes = np.random.uniform(10, 500, size=100)

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": prices,
            "high": prices + 50,
            "low": prices - 50,
            "close": prices,
            "volume": volumes,
        })

        featured = app.build_features(df)

        expected_cols = [
            "log_return", "vol_4h", "vol_24h", "vol_72h",
            "return_1h", "return_4h", "return_24h",
            "volume_change_24h", "range_pct"
        ]
        for col in expected_cols:
            self.assertIn(col, featured.columns)

        # Non-null values after 72 rows
        valid_rows = featured.dropna(subset=expected_cols)
        self.assertGreater(len(valid_rows), 0)

    def test_model_bundles_load(self):
        """Verify that saved model bundles load correctly and have required keys."""
        vol_bundle = app.load_volatility_model()
        self.assertIn("model", vol_bundle)
        self.assertIn("features", vol_bundle)
        self.assertTrue(hasattr(vol_bundle["model"], "predict"))

        dir_bundle = app.load_direction_model()
        self.assertIn("model", dir_bundle)
        self.assertIn("features", dir_bundle)
        self.assertTrue(hasattr(dir_bundle["model"], "predict_proba"))

    def test_db_manager_lifecycle(self):
        """Verify observation save, retrieval, and stats calculation."""
        # Use a temporary local test file
        test_file = BASE_DIR / "data" / "observations_test.json"
        db_manager.LOCAL_BACKUP_PATH = test_file

        try:
            doc = {
                "_id": "test-obs-123",
                "initial_price": 80000.0,
                "horizon_hours": 4,
                "ai_forecast": {"direction": "UP", "probability_up": 55.0},
                "user_prediction": {"direction": "UP", "notes": "Unit test observation"},
                "status": "PENDING"
            }

            ok, msg, saved = db_manager.save_observation(doc)
            self.assertTrue(ok)

            all_obs = db_manager.get_all_observations()
            self.assertEqual(len(all_obs), 1)
            self.assertEqual(all_obs[0]["_id"], "test-obs-123")

            # Update evaluation
            db_manager.update_observation_evaluation(
                "test-obs-123",
                {"actual_price": 81000.0, "actual_direction": "UP", "ai_correct": True, "user_correct": True},
                new_status="COMPLETED"
            )

            stats = db_manager.calculate_accuracy_stats()
            self.assertEqual(stats["completed_count"], 1)
            self.assertEqual(stats["ai_accuracy"], 100.0)
            self.assertEqual(stats["user_accuracy"], 100.0)

        finally:
            if test_file.exists():
                test_file.unlink()
            # Reset to original path
            db_manager.LOCAL_BACKUP_PATH = BASE_DIR / "data" / "observations_backup.json"


if __name__ == "__main__":
    unittest.main()
