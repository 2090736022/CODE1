import unittest
from unittest.mock import Mock, patch

import pandas as pd
import chaoduan as strategy


class VolumeRelaxationTests(unittest.TestCase):
    def daily(self, volume=201):
        return pd.DataFrame({
            "trade_date": ["20200102", "20200103", "20200106", "20200107", "20200108", "20200109"],
            "vol": [100] * 5 + [volume], "close": [10] * 6,
        })

    def test_ratio_and_invalid_history(self):
        daily = self.daily()
        self.assertAlmostEqual(strategy.observation_volume_ratio(daily.iloc[::-1], "20200109"), 201 / ((400 + 201) / 5))
        self.assertAlmostEqual(strategy.observation_volume_ratio(daily.iloc[1:], "20200109"), 201 / ((400 + 201) / 5))
        extra = pd.concat([daily, pd.DataFrame({"trade_date": ["20200110"], "vol": [99999]})])
        self.assertEqual(strategy.observation_volume_ratio(extra, "20200109"), strategy.observation_volume_ratio(daily, "20200109"))
        for invalid in [daily.iloc[2:], daily.iloc[:-1], daily.assign(vol=0),
                        daily.assign(vol=[100, None, 100, 100, 100, 201])]:
            with self.assertRaises(ValueError):
                strategy.observation_volume_ratio(invalid, "20200109")

    def test_boundaries_and_search_integration(self):
        basic = pd.DataFrame({"ts_code": ["000001.SZ"], "turnover_rate_f": [10], "circ_mv": [1000000]})
        for daily_volume, minute_volume, expected, threshold in [
            (200, 340, False, 4), (201, 340, True, 3.4),
            (201, 339, False, 3.4), (200, 400, True, 4),
        ]:
            daily = self.daily(daily_volume)
            daily.loc[daily.trade_date < "20200109", "vol"] = 75
            observed = pd.DataFrame({"datetime": pd.date_range("2020-01-09 09:45", periods=2, freq="min"),
                                     "cumulative_volume": [minute_volume] * 2, "price": [10.1] * 2})
            selected = pd.DataFrame({"datetime": pd.date_range("2020-01-08 09:45", periods=2, freq="min"),
                                     "cumulative_volume": [100] * 2})
            with self.subTest(daily_volume=daily_volume, minute_volume=minute_volume), \
                 patch.object(strategy, "get_next_trade_date", return_value="20200109"), \
                 patch.object(strategy, "get_daily_basic", return_value=basic), \
                 patch.object(strategy.pd, "read_csv", return_value=basic), \
                 patch.object(strategy, "filter_non_st_stocks", side_effect=lambda df: df), \
                 patch.object(strategy, "get_daily_data", return_value=(daily, False)), \
                 patch.object(strategy, "get_cumulative_window", side_effect=[observed, selected]), \
                 patch.object(strategy, "max_volume_minute_is_bullish", return_value=True), \
                 patch.object(strategy, "show_progress"):
                result = strategy.run_search(as_of_date="20200108", pro=Mock(), save_file=False,
                                             daily_matcher=lambda df: True, basic_matcher=lambda row, code: True)
                self.assertEqual(len(result), int(expected))
                if expected:
                    self.assertEqual(result.iloc[0]["minute_volume_threshold"], threshold)
                    self.assertEqual(result.iloc[0]["observation_daily_volume_ratio"], daily_volume / ((300 + daily_volume) / 5))


if __name__ == "__main__":
    unittest.main()
