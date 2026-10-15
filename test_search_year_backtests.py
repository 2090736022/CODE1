"""验证 test/turn 年度回测与原策略的对应关系。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import botton_year_backtest as engine
from year_backtest_strategies import StrategySpec


class SearchBacktestTests(unittest.TestCase):
    def test_both_source_return_functions(self):
        df = pd.DataFrame({"ts_code": ["TEST.SZ"] * 5, "trade_date": [f"2026010{i}" for i in range(1, 6)],
                           "open": [10, 8, 6, 7, 9], "close": [10, 9, 7, 11, 10]})
        for name in ("test", "turn"):
            source = StrategySpec(name).module
            with tempfile.TemporaryDirectory(dir=engine.ROOT) as temp:
                df.to_csv(Path(temp) / "TEST_SZ.csv", index=False)
                with patch.object(source, "DAILY_CACHE_DIR", Path(temp)), patch.object(source, "BACKTEST_FORWARD_DAYS", 4):
                    expected = source.calculate_forward_5d_return("TEST.SZ", "20260101")
                actual = engine.best_forward_result(df.iloc[1:], 4)
                for key, value in expected.items():
                    self.assertEqual(actual[key], value)

    def test_basic_thresholds_and_duplicate_test_rows(self):
        for name in ("test", "turn"):
            spec = StrategySpec(name)
            history = pd.DataFrame({"pct_chg": [4]})
            row = {"ts_code": "TEST.SZ", "turnover_rate": spec.min_turnover_rate,
                   "total_mv": spec.min_total_mv + 1}
            self.assertFalse(spec.basic_matches(pd.DataFrame([row]), "TEST.SZ", "20260101", history))
            row["turnover_rate"] += .001
            self.assertTrue(spec.basic_matches(pd.DataFrame([row]), "TEST.SZ", "20260101", history))
            row["total_mv"] = spec.min_total_mv
            self.assertFalse(spec.basic_matches(pd.DataFrame([row]), "TEST.SZ", "20260101", history))
        spec = StrategySpec("test")
        rows = pd.DataFrame({"ts_code": ["TEST.SZ", "TEST.SZ"], "turnover_rate": [5, 11],
                             "total_mv": [spec.min_total_mv+1]*2})
        self.assertTrue(spec.basic_matches(rows, "TEST.SZ", "20260101", pd.DataFrame({"pct_chg": [9.86]})))

    def test_adapter_against_exhaustive_source_dates(self):
        # 覆盖一年内所有日期，包括被预筛排除的日期，以检查漏选。
        for name, code in (("test", "000887_SZ"), ("turn", "000811_SZ")):
            spec = StrategySpec(name)
            path = engine.ROOT / f"model_artifacts/download_daily/daily/{code}.csv"
            if not path.exists():
                self.skipTest("local sample unavailable")
            df = spec.module.normalize_daily_frame(pd.read_csv(path, dtype={"trade_date": str}))
            df = df.sort_values("trade_date").reset_index(drop=True)
            mask = spec.candidate_mask(df)
            pd.testing.assert_series_equal(spec.candidate_mask(df.iloc[:-10]), mask.iloc[:-10])
            indices = df.index[df.trade_date.between("20251015", "20261014")]
            for i in indices:
                day = df.iloc[i].trade_date
                lower = str(int(day) - engine.config.SEARCH_DAILY_LOOKBACK_YEARS * 10000)
                h = df.iloc[:i+1]
                h = h[h.trade_date >= lower].copy()
                expected = ["main"] if spec.module.match_strategy(h) else []
                if name == "test" and spec.module.match_volume_price_strategy(h):
                    expected.append("volume_price")
                actual = spec.match(h) if mask.iloc[i] else []
                self.assertEqual(actual, expected, (name, day))


if __name__ == "__main__":
    unittest.main()
