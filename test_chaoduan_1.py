import unittest
from unittest.mock import patch
import pandas as pd
import chaoduan
import chaoduan_1 as strategy


class StrategyTests(unittest.TestCase):
    def test_selection_day_one_price_limit_down(self):
        frame = pd.DataFrame({"trade_date": pd.date_range("2026-08-01", periods=20),
                              "close": [5] * 10 + [10] * 9 + [9],
                              "open": 9.0, "high": 9.0, "low": 9.0,
                              "pre_close": 10, "pct_chg": -10})
        self.assertFalse(strategy.match_strategy(frame.iloc[::-1]))
        frame.loc[19, "high"] = 9.1
        self.assertTrue(strategy.match_strategy(frame))
        row = dict(open=9.01, high=9.01, low=9.01, close=9.01, pre_close=10.01)
        self.assertTrue(strategy.is_one_price_limit_down(row))
        row["pre_close"] = 10
        self.assertFalse(strategy.is_one_price_limit_down(row))

    def test_strict_basic_limits(self):
        for mv, turnover, expected in [(200000, 3, False), (200001, 1.25, False),
                                       (200001, 1.26, True), (500000, None, False),
                                       (20000000, 45, True)]:
            self.assertEqual(strategy.basic_matches({"circ_mv": mv,
                "turnover_rate_f": turnover, "turnover_rate": 10}, "000001.SZ"), expected)

    def test_either_moving_average_condition(self):
        for prices, expected in [([10]*15+[11]*5, True),
                                 ([5]*10+[12]*5+[10]*5, True),
                                 ([10]*20, False), (list(range(20, 0, -1)), False),
                                 ([10]*19, False), ([10]*19+[float("nan")], False)]:
            frame = pd.DataFrame({"trade_date": pd.date_range("2026-08-01", periods=len(prices)), "close": prices})
            self.assertEqual(strategy.match_strategy(frame), expected)

    def test_extended_window_and_four_times_volume(self):
        selected = pd.DataFrame({"datetime": pd.date_range("2026-09-16 09:45", periods=46, freq="min"),
                                 "cumulative_volume": [100] * 46})
        for last_volume, expected in [(399, False), (400, True)]:
            observed = pd.DataFrame({"datetime": pd.date_range("2026-09-17 09:45", periods=46, freq="min"),
                                     "cumulative_volume": [399] * 45 + [last_volume], "price": [11] * 46})
            with patch.object(chaoduan, "get_cumulative_window", side_effect=[observed, selected]) as fetch:
                result = strategy.minute_volume_matches("000001.SZ", "20260917", "20260916", previous_close=10)
                self.assertEqual(result[0], expected)
                self.assertEqual(result[3], "10:30:00")
                self.assertEqual(fetch.call_args_list[0].args[2:], ("09:45:00", "10:30:00"))

    def test_reuses_minute_strategy_and_separate_output(self):
        self.assertIs(strategy.minute_volume_matches, chaoduan.minute_volume_matches)
        with patch.object(strategy, "run_base_search") as run:
            strategy.run_search(as_of_date="20260916", save_file=False)
            self.assertIs(run.call_args.kwargs["daily_matcher"], strategy.match_strategy)
            self.assertIs(run.call_args.kwargs["basic_matcher"], strategy.basic_matches)
            self.assertEqual(run.call_args.kwargs["result_prefix"], "超短1筛选结果")

    def test_growth_jump_rejects_entire_window(self):
        cases = [
            ([130, 130, 340] + [400]*28, False),  # 30% -> 240%, then qualifies
            ([130, 130, 330] + [400]*28, True),   # exactly 200 percentage points
            ([400]*30 + [601], False),          # jump at 10:15 after earlier signal
            ([400]*31, True),                   # stable 300% is allowed
        ]
        for volumes, expected in cases:
            observed = pd.DataFrame({"datetime": pd.date_range("2026-09-17 09:45", periods=31, freq="min"),
                                     "cumulative_volume": volumes, "price": [11]*31})
            selected = pd.DataFrame({"datetime": pd.date_range("2026-09-16 09:45", periods=31, freq="min"),
                                     "cumulative_volume": [100]*31})
            with self.subTest(volumes=volumes), patch.object(chaoduan, "get_cumulative_window", side_effect=[observed, selected]):
                self.assertEqual(strategy.minute_volume_matches("000001.SZ", "20260917", "20260916", previous_close=10)[0], expected)

    def test_selection_day_cannot_be_limit_up(self):
        frame = pd.DataFrame({"trade_date": pd.date_range("2026-08-01", periods=20),
                              "close": [10] * 19 + [11], "pct_chg": [10] * 19 + [9.5]})
        self.assertTrue(strategy.match_strategy(frame))
        frame.loc[19, "pct_chg"] = 9.51
        self.assertFalse(strategy.match_strategy(frame.iloc[::-1]))
        frame.loc[19, "pct_chg"] = float("nan")
        frame.loc[19, "pre_close"] = 10
        self.assertFalse(strategy.match_strategy(frame))

    def test_early_limit_up_including_intraminute_touch(self):
        for maximum, close, ratio, expected in [(10.99, 11, 1.5, True),
                (11, 11, 2.1, False), (11, 10.99, 1.5, True),
                (11, 11, 2.1001, True), (11, 11, 2, False)]:
            with patch.object(strategy, "get_cumulative_window") as opening_fetch, \
                 patch.object(strategy, "get_window", return_value=pd.DataFrame({"datetime": ["2026-09-17 09:39:00"], "high": [maximum], "close": [10]})) as fetch:
                self.assertEqual(strategy.early_limit_up_filter("000001.SZ", "20260917", 10,
                    observation_close=close, daily_volume_ratio=ratio), expected)
                opening_fetch.assert_not_called()
                fetch.assert_called_once_with("000001.SZ", "20260917", "09:31:00", "09:39:00")


if __name__ == "__main__":
    unittest.main()
