import unittest
from unittest.mock import Mock, patch

import pandas as pd

import chaoduan


class NextMorningStrategyTests(unittest.TestCase):
    def test_calendar_skips_weekend_and_holiday(self):
        pro = Mock()
        pro.trade_cal.return_value = pd.DataFrame({
            "cal_date": ["20261009", "20260930"], "is_open": [1, 1],
        })
        self.assertEqual(chaoduan.get_next_trade_date(pro, "20260930"), "20261009")

    def test_cumulative_volume_hits_at_0946_including_exact_double(self):
        for volume, expected in [(200000, True), (199999, False), (0, False)]:
            observed = pd.DataFrame({"datetime": pd.date_range("2026-09-17 09:45", periods=16, freq="min"),
                                     "cumulative_volume": [100000, volume] + [200001] * 14})
            selected = pd.DataFrame({"datetime": pd.date_range("2026-09-16 09:45", periods=16, freq="min"),
                                     "cumulative_volume": [100000, 100000] + [150000] * 14})
            observed["price"] = 10
            with self.subTest(volume=volume), patch.object(
                chaoduan, "get_cumulative_window", side_effect=[observed, selected]
            ) as fetch:
                result = chaoduan.minute_volume_matches("000001.SZ", "20260917", "20260916", previous_close=10)
                self.assertEqual(result[0], expected)
                if expected:
                    self.assertEqual(result[1:], (200000, 100000, "09:46:00", 0))
                self.assertEqual(fetch.call_args_list[0].args[1], "20260917")
                self.assertEqual(fetch.call_args_list[1].args[1], "20260916")

    def test_minute_volume_ignores_other_dates(self):
        data = pd.DataFrame({
            "时间": ["2026-09-16 09:45:00", "2026-09-17 09:45:00", "2026-09-17 10:01:00"],
            "成交量": [999, 100, 999],
        })
        self.assertEqual(chaoduan._minute_volume(data, "20260917"), 100)

    def test_search_uses_selection_day_then_next_morning(self):
        pro = Mock()
        pro.trade_cal.return_value = pd.DataFrame({
            "cal_date": ["20260916", "20260917"], "is_open": [1, 1],
        })
        basic = pd.DataFrame({
            "ts_code": ["000001.SZ"], "turnover_rate_f": [10], "circ_mv": [1_000_000],
        })
        # Future rows must not leak into daily selection; stale rows must not stand in for it.
        for last_date, expected_count in [("20260916", 1), ("20260915", 0)]:
            daily = pd.DataFrame({"trade_date": [last_date, "20260917"], "close": [10, 11]})
            with self.subTest(last_date=last_date), \
                    patch.object(chaoduan.pd, "read_csv", return_value=basic), \
                    patch.object(chaoduan, "filter_non_st_stocks", side_effect=lambda df: df), \
                    patch.object(chaoduan, "get_daily_basic", return_value=basic), \
                    patch.object(chaoduan, "get_daily_data", return_value=(daily, False)) as get_daily, \
                    patch.object(chaoduan, "match_strategy", return_value=True) as match, \
                    patch.object(chaoduan, "minute_volume_matches", return_value=(True, 200, 100, "09:46:00", 0)) as minute, \
                    patch.object(chaoduan, "show_progress"):
                result = chaoduan.run_search(as_of_date="20260916", pro=pro, save_file=False)
                self.assertEqual(len(result), expected_count)
                self.assertEqual(get_daily.call_args.args[-1], "20260916")
                if expected_count:
                    self.assertEqual(match.call_args.args[0]["trade_date"].tolist(), ["20260916"])
                    minute.assert_called_once_with("000001.SZ", "20260917", "20260916", previous_close=10)
                    self.assertEqual(result.iloc[0]["trade_date"], "20260916")
                    self.assertEqual(result.iloc[0]["minute_trade_date"], "20260917")
                    self.assertEqual(result.iloc[0]["previous_trade_date"], "20260916")
                    self.assertEqual(result.iloc[0]["minute_match_time"], "09:46:00")
                    self.assertEqual(result.iloc[0]["cumulative_volume"], 200)
                else:
                    minute.assert_not_called()

    def test_price_and_volume_must_match_same_minute(self):
        for prices, expected, clock in [([9.9] * 16, False, "10:00:00"),
                                       ([9.8, 9.91] + [10] * 14, True, "09:46:00")]:
            observed = pd.DataFrame({"datetime": pd.date_range("2026-09-17 09:45", periods=16, freq="min"),
                                     "cumulative_volume": [200] * 16, "price": prices})
            selected = pd.DataFrame({"datetime": pd.date_range("2026-09-16 09:45", periods=16, freq="min"),
                                     "cumulative_volume": [100] * 16})
            with patch.object(chaoduan, "get_cumulative_window", side_effect=[observed, selected]):
                result = chaoduan.minute_volume_matches("000001.SZ", "20260917", "20260916", previous_close=10)
                self.assertEqual((result[0], result[3]), (expected, clock))


if __name__ == "__main__":
    unittest.main()
