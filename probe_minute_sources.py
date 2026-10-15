"""Read-only live checks of public one-minute quote endpoints."""
from concurrent.futures import ThreadPoolExecutor
import requests

TARGETS = {
    "eastmoney_trends": ("https://push2his.eastmoney.com/api/qt/stock/trends2/get", {"secid": "0.000001", "ndays": "5", "iscr": "0", "fields1": "f1,f2,f3", "fields2": "f51,f52,f53,f54,f55,f56,f57,f58"}),
    "tencent_web": ("https://web.ifzq.gtimg.cn/appstock/app/kline/mkline", {"param": "sz000001,m1,,640"}),
    "tencent_web2": ("https://web2.ifzq.gtimg.cn/appstock/app/kline/mkline", {"param": "sz000001,m1,,640"}),
    "tencent_web3": ("https://web3.ifzq.gtimg.cn/appstock/app/kline/mkline", {"param": "sz000001,m1,,640"}),
    "tencent_days": ("https://web.ifzq.gtimg.cn/appstock/app/day/query", {"code": "sz000001"}),
    "sina": ("https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_data=/CN_MarketDataService.getKLineData", {"symbol": "sz000001", "scale": "1", "ma": "no", "datalen": "1023"}),
}

def probe(item):
    name, (url, params) = item
    try:
        response = requests.get(url, params=params, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        return name, response.status_code, response.url, response.text[:350], response.text[-250:]
    except Exception as error:
        return name, str(error)

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=6) as pool:
        for result in pool.map(probe, TARGETS.items()):
            print(result, flush=True)
