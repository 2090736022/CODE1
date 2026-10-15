import pandas as pd


def filter_non_st_stocks(stock_df):
    data = stock_df.copy()
    if "name" in data.columns:
        non_st_mask = ~data["name"].astype(str).str.contains("ST", case=False, na=False)
        data = data[non_st_mask].copy()
    return data


def filter_non_st_codes(stock_codes, stock_file):
    stock_meta = pd.read_csv(stock_file, dtype={"ts_code": str})
    stock_meta = filter_non_st_stocks(stock_meta)
    if "ts_code" not in stock_meta.columns:
        return list(stock_codes)

    allowed_codes = set(stock_meta["ts_code"].dropna().astype(str).str.strip())
    return [code for code in stock_codes if str(code).strip() in allowed_codes]
