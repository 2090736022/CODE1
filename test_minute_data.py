import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

import pandas as pd
import minute_data


class MinuteDataTests(unittest.TestCase):
    def test_full_day_refreshes_incomplete_cache_and_preserves_on_failure(self):
        times = pd.date_range("2026-09-18 09:31", periods=120, freq="min").append(
            pd.date_range("2026-09-18 13:01", periods=117, freq="min")).append(
            pd.DatetimeIndex(["2026-09-18 15:00"]))
        complete = pd.DataFrame({"datetime": times, "volume": 10})
        partial = complete.iloc[:-1]
        with TemporaryDirectory() as directory, patch.object(minute_data, "CACHE_DIR", Path(directory)):
            cache = Path(directory) / "603001_SH" / "20260918.csv"
            cache.parent.mkdir()
            partial.to_csv(cache, index=False)
            with patch.object(minute_data, "fetch_recent_minutes", side_effect=[partial, complete]) as fetch:
                self.assertEqual(len(minute_data.get_full_day("603001.SH", "20260918")), 238)
                self.assertEqual(fetch.call_count, 2)
                self.assertEqual(len(pd.read_csv(cache)), 238)
            with patch.object(minute_data, "fetch_recent_minutes") as fetch:
                self.assertEqual(len(minute_data.get_full_day("603001.SH", "20260918")), 238)
                fetch.assert_not_called()
            partial.to_csv(cache, index=False)
            original = cache.read_bytes()
            with patch.object(minute_data, "fetch_recent_minutes", return_value=partial) as fetch:
                with self.assertRaises(ValueError):
                    minute_data.get_full_day("603001.SH", "20260918")
                self.assertEqual(fetch.call_count, 2)
                self.assertEqual(cache.read_bytes(), original)

    def test_missing_bar_is_error(self):
        data = pd.DataFrame({"datetime": pd.date_range("2026-09-16 09:45", periods=15, freq="min")})
        with self.assertRaises(ValueError):
            minute_data.window(data, "20260916", "09:45:00", "10:00:00")

    def test_cache_reuses_both_dates_and_old_date_fails(self):
        data = pd.DataFrame({"datetime": list(pd.date_range("2026-09-16 09:45", periods=16, freq="min")) + list(pd.date_range("2026-09-17 09:45", periods=16, freq="min")), "volume": [10] * 32})
        with TemporaryDirectory() as directory, patch.object(minute_data, "CACHE_DIR", Path(directory)), patch.object(minute_data, "fetch_recent_minutes", return_value=data) as fetch:
            self.assertEqual(minute_data.get_window("000001.SZ", "20260916")["volume"].sum(), 160)
            self.assertEqual(minute_data.get_window("000001.SZ", "20260917")["volume"].sum(), 160)
            fetch.assert_called_once()
            with self.assertRaises(ValueError):
                minute_data.get_window("000001.SZ", "20200101")


if __name__ == "__main__":
    unittest.main()
