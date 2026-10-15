from pathlib import Path
import pandas as pd

root=Path('model_artifacts/minute_history/raw_stockdb')
for code,market in [('000001','SZ'),('600000','SH')]:
    frames=[pd.read_csv(p,dtype={'code':str}) for p in root.glob(f'{code}_*.csv')]
    data=pd.concat(frames).drop_duplicates('datetime').sort_values('datetime')
    daily=pd.read_csv(f'model_artifacts/download_daily/daily/{code}_{market}.csv',dtype={'trade_date':str}).set_index('trade_date')
    for day,frame in data.groupby(data.datetime.str[:10]):
        date=day.replace('-','')
        total=frame.volume.sum()/100
        expected=pd.date_range(day+' 09:30',day+' 10:15',freq='min')
        missing=expected.difference(pd.to_datetime(frame.datetime))
        daily_vol=daily.loc[date,'vol'] if date in daily.index else 0
        print(code,date,'bars',len(frame),'morning_missing',len(missing),'sum',round(total,2),'daily',daily_vol,'error_pct',round((total/daily_vol-1)*100,3) if daily_vol else None,flush=True)
    cached=Path(f'model_artifacts/minute_tencent_cumulative/{code}_{market}/20260914.csv')
    if cached.exists():
        recent=data[data.datetime.str.startswith('2026-09-14')].copy()
        recent['history_cum']=recent.volume.cumsum()/100
        comparison=recent.merge(pd.read_csv(cached),on='datetime')
        comparison=comparison[comparison.datetime.str[11:].between('09:45:00','10:15:00')]
        print('CUMULATIVE_COMPARE',code,comparison[['datetime','history_cum','cumulative_volume']].to_string(index=False),flush=True)
