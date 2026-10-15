import unittest
from unittest.mock import patch

import pandas as pd
import chaoduan
from minute_data import full_day


class MaxVolumeMinuteTests(unittest.TestCase):
    def bars(self):
        times = pd.date_range("2020-01-09 09:31", periods=120, freq="min").append(
            pd.date_range("2020-01-09 13:01", periods=117, freq="min")).append(
            pd.DatetimeIndex(["2020-01-09 15:00"]))
        return pd.DataFrame({"datetime": times, "open": 10.0, "high": 10.1, "close": 10.1, "volume": 1.0})

    def check(self, bars, previous_close=None):
        with patch.object(chaoduan, "get_full_day", side_effect=lambda code, day: full_day(bars, day)):
            return chaoduan.max_volume_minute_is_bullish("000001.SZ", "20200109", previous_close=previous_close)

    def test_down_bar_high_at_limit_up_is_allowed(self):
        bars = self.bars()
        bars.loc[0, ["open", "high", "close", "volume"]] = [10.8, 11, 10.5, 100]
        self.assertTrue(self.check(bars, previous_close=10))
        bars.loc[0, "high"] = 10.99
        self.assertFalse(self.check(bars, previous_close=10))
        bars.loc[0, "high"] = 11.01
        self.assertFalse(self.check(bars, previous_close=10))
        self.assertTrue(self.check(bars, previous_close=10.01))
        bars.loc[237, ["open", "close", "volume"]] = [10, 9.9, 100]
        self.assertFalse(self.check(bars, previous_close=10.01))

    def test_maximum_anywhere_in_full_day(self):
        for position in [0, 119, 120, 236, 237]:
            for close, expected in [(10.1, True), (10, True), (9.9, False)]:
                bars = self.bars()
                bars.loc[position, ["volume", "close"]] = [100, close]
                with self.subTest(position=position, close=close):
                    self.assertEqual(self.check(bars), expected)

    def test_tied_maxima_must_all_be_bullish(self):
        bars = self.bars()
        bars.loc[[0, 237], "volume"] = 100
        self.assertTrue(self.check(bars))
        bars.loc[0, "close"] = 10
        self.assertTrue(self.check(bars))
        bars.loc[0, "close"] = 9.9
        self.assertFalse(self.check(bars))

    def test_missing_or_invalid_data_rejected(self):
        bars = self.bars()
        for invalid in [bars.iloc[:-1], bars.drop(index=120), bars.assign(volume=0),
                        bars.assign(close=float("nan")), bars.assign(high=float("nan")), bars.assign(volume=float("inf"))]:
            with self.assertRaises(ValueError):
                self.check(invalid)


if __name__ == "__main__":
    unittest.main()
