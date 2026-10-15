import unittest
import pandas as pd
import download


class DownloadCacheTests(unittest.TestCase):
    def test_cache_coverage_normalizes_dates(self):
        data = pd.DataFrame({"trade_date": ["20200101", "20200102", "20200103"]})
        self.assertTrue(download.cache_covers(data, "20200101", "20200103"))
        self.assertTrue(download.cache_covers(data, "20200102", "20200103"))
        self.assertFalse(download.cache_covers(data, "20191231", "20200103"))
        self.assertFalse(download.cache_covers(data.iloc[:2], "20200101", "20200103"))

    def test_completed_download_skips_api(self):
        cached = pd.DataFrame({"trade_date": ["20200101", "20200103"], "close": [1, 2]})
        class Pro:
            def daily(self, **kwargs):
                raise AssertionError("cache hit must not call API")
        old = download.DAILY_CACHE_DIR
        try:
            from tempfile import TemporaryDirectory
            with TemporaryDirectory() as directory:
                download.DAILY_CACHE_DIR = pd.io.common.stringify_path(directory)
                from pathlib import Path
                download.DAILY_CACHE_DIR = Path(directory)
                download.DAILY_CACHE_DIR.mkdir(exist_ok=True)
                cached.to_csv(download.DAILY_CACHE_DIR / "000001_SZ.csv", index=False)
                self.assertFalse(download.download_daily_for_stock(Pro(), "000001.SZ", "20200101", "20200103"))
        finally:
            download.DAILY_CACHE_DIR = old


if __name__ == "__main__":
    unittest.main()
