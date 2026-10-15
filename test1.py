import json
import requests
from loguru import logger


def _get_market_code(stock_code):
    """
    根据股票代码计算出市场代码。
    :param stock_code: 股票代码
    :return: 市场代码，0：深圳，1：上海，2：其他
    """
    # 获取股票代码的前缀
    code_prefix = int(stock_code[:3])

    # 根据前缀判断市场
    if code_prefix in [0, 2, 3]:  # 深圳股票代码前缀一般为 0、2、3
        return 0  # 深圳市场
    elif code_prefix in [6, 9]:  # 上海股票代码前缀一般为 6、9
        return 1  # 上海市场
    else:
        return 2  # 其他市场，此处假设为北京市场


def _handle_event(event_data) -> str:
    """
    解析event数据
    :param event_data: SSE数据
    :return: 事件的数据字段
    """
    lines = event_data.strip().split('\n')
    data = ""

    for line in lines:
        if line.startswith("event:"):  # 东财分时数据的事件无event
            event_type = line.replace("event:", "").strip()
        elif line.startswith("data:"):
            data = line.replace("data:", "").strip()
    # logger.info(f"data received: {data}")
    return data


def get_minutely_data(code: str) -> dict:
    """
    获取最新的分时数据
    :param code: 股票代码，6位数字格式，比如：“000001”。
    :return: 数据字典
    """
    # url格式，需要调用者依次填充：市场代码（0：深圳，0：上海），股票代码，日期窗口
    url_format = "https://45.push2.eastmoney.com/api/qt/stock/trends2/sse?fields1=f1,f2,f3,f4,f5,f6,f7,f8," \
                 "f9,f10,f11,f12,f13,f14,f17&fields2=f51,f52,f53,f54,f55,f56,f57," \
                 "f58&mpi=1000&ut=fa5fd1943c7b386f172d6893dbfba10b&secid={market}.{code}" \
                 "&ndays={days}&iscr=0&iscca=0&wbp2u=1849325530509956|0|1|0|web"

    url = url_format.format(market=_get_market_code(code), code=code, days=1)  # 请求url
    print(url)
    headers = {
        "Accept": "text/event-stream"
    }

    response = requests.get(url, headers=headers, stream=True)  # 请求数据，开启流式传输

    if response.status_code == 200:
        event_data = ""
        for chunk in response.iter_content(chunk_size=None):  # 不断地获取数据
            data_decoded = chunk.decode('utf-8')  # 将字节流解码成字符串
            event_data += data_decoded
            parts = event_data.split('\n\n')
            if len(parts) > 1:  # 获取到一条完整的事件后，对数据进行解析。
                data = _handle_event(parts[0])
                return json.loads(data)


if __name__ == "__main__":  # 测试代码
    data_test = get_minutely_data("000001")
    logger.info(f"data_test: {data_test}")