import unittest
from unittest.mock import Mock, patch
import pandas as pd
import chaoduan


class ObservationTurnoverTests(unittest.TestCase):
    def test_closing_gain_is_strict_and_rejections_are_silent(self):
        basic = pd.DataFrame({"ts_code": ["000001.SZ"], "turnover_rate_f": [9], "circ_mv": [500000]})
        selected = pd.DataFrame({"trade_date": ["20200106"], "close": [10]})
        for gain, count in [(-1, 0), (0, 0), (0.01, 1)]:
            closing = pd.DataFrame({"trade_date": ["20200107"], "pct_chg": [gain]})
            with self.subTest(gain=gain), \
                 patch.object(chaoduan, "get_next_trade_date", return_value="20200107"), \
                 patch.object(chaoduan, "get_daily_basic", return_value=basic), \
                 patch.object(chaoduan.pd, "read_csv", return_value=basic), \
                 patch.object(chaoduan, "get_daily_data", side_effect=[(selected, False), (closing, False)]) as fetch, \
                 patch.object(chaoduan, "minute_volume_matches", return_value=(True, 400, 100, "09:50:00", 1)) as minute, \
                 patch("builtins.print") as output:
                result = chaoduan.run_search(as_of_date="20200106", pro=Mock(), save_file=False,
                    daily_matcher=lambda df: True, basic_matcher=lambda row, code: True, observation_positive_close=True)
                self.assertEqual(len(result), count)
                self.assertEqual(fetch.call_args.args[-2:], ("20200107", "20200107"))
                if not count:
                    minute.assert_not_called()
                    self.assertFalse(any("000001.SZ" in str(call) for call in output.call_args_list))

    def test_observation_day_strict_threshold_and_output(self):
        selection = pd.DataFrame({"ts_code": ["000001.SZ"], "turnover_rate_f": [3], "circ_mv": [500000]})
        daily = pd.DataFrame({"trade_date": ["20200106"], "close": [10]})
        for value, count in [(8, 0), (8.01, 1), (None, 0)]:
            observation = selection.copy()
            observation["turnover_rate_f"] = value
            with self.subTest(value=value), \
                 patch.object(chaoduan, "get_next_trade_date", return_value="20200107"), \
                 patch.object(chaoduan, "get_daily_basic", side_effect=[observation, selection]) as basic, \
                 patch.object(chaoduan.pd, "read_csv", return_value=selection), \
                 patch.object(chaoduan, "get_daily_data", return_value=(daily, False)), \
                 patch.object(chaoduan, "minute_volume_matches", return_value=(True, 400, 100, "09:50:00", 1)):
                result = chaoduan.run_search(as_of_date="20200106", pro=Mock(), save_file=False,
                    daily_matcher=lambda df: True, basic_matcher=lambda row, code: True, observation_min_turnover=8)
                self.assertEqual(len(result), count)
                self.assertEqual(basic.call_args_list[0].args[1], "20200107")
                if count:
                    self.assertEqual(result.iloc[0]["observation_turnover_rate_f"], 8.01)
                    self.assertEqual(result.iloc[0]["turnover_rate_f"], 3)


if __name__ == "__main__":
    unittest.main()
