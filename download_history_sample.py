"""下载公开体验接口的少量历史样本，仅验证用，禁止全市场循环调用。"""
import sys
from pathlib import Path
import requests
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parent / '.vendor' / 'msgpack'))
import msgpack


def download(code, month):
    response = requests.get('http://8.138.149.215:7899/',
        params={'cmd': 'get', 't': f'分钟k:{code}:{month}*'}, timeout=30)
    response.raise_for_status()
    payload = msgpack.unpackb(response.content, raw=False)
    print('response',code,type(payload).__name__,len(payload),str(payload)[:450],flush=True)
    if isinstance(payload, dict):
        records = list(payload.values()) if 'date' not in payload else [payload]
    else:
        records = [item[1] if isinstance(item, list) and len(item)==2 else item for item in payload]
    data = pd.DataFrame(records)
    if 'date' not in data:
        raise ValueError('No date column')
    data['datetime'] = pd.to_datetime(data['date'].astype(str),format='%Y%m%d%H%M%S')
    folder=Path('model_artifacts/minute_history/raw_stockdb')
    folder.mkdir(parents=True,exist_ok=True)
    data.to_csv(folder/f'{code}_{month}.csv',index=False,encoding='utf-8-sig')
    counts=data.groupby(data.datetime.dt.strftime('%Y%m%d')).size()
    print(code,'rows',len(data),'first',data.datetime.min(),'last',data.datetime.max(),'days',len(counts),'counts',counts.to_dict(),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    download('000001', '202609')
    download('600000', '202608')
    download('600000', '202609')
