from concurrent.futures import ThreadPoolExecutor
import requests
import sys
sys.stdout.reconfigure(encoding='utf-8')

targets = {
 'mootdx_consts': ('https://raw.githubusercontent.com/mootdx/mootdx/master/mootdx/consts.py', {}),
 'search_data': ('https://api.github.com/search/repositories', {'q':'stock minute data china','per_page':8}),
 'search_1min': ('https://api.github.com/search/repositories', {'q':'股票 分钟 数据','sort':'stars','per_page':8}),
 'sina_large': ('https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_data=/CN_MarketDataService.getKLineData', {'symbol':'sz000001','scale':'1','ma':'no','datalen':'10000'}),
 'sina_date': ('https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_data=/CN_MarketDataService.getKLineData', {'symbol':'sz000001','scale':'1','ma':'no','datalen':'1023','end_date':'2026-08-18'}),
 'github_search': ('https://api.github.com/search/repositories', {'q':'A股 1分钟 数据','sort':'updated','per_page':5}),
 'mootdx_hosts': ('https://raw.githubusercontent.com/mootdx/mootdx/master/mootdx/server.py', {}),
 'tdx_http': ('https://data.tdx.com.cn/tdx/history/base/hq/sh/20260917/1min.zip', {}),
}

def probe(item):
 name,(url,params)=item
 try:
  r=requests.get(url,params=params,timeout=18,headers={'User-Agent':'Mozilla/5.0'})
  if name.startswith('search') or name=='github_search':
   print(name,r.status_code,[(x['full_name'],str(x.get('description'))[:180]) for x in r.json().get('items',[])],flush=True)
  else:
   print(name,r.status_code,len(r.content),r.text[:14000] if name=='mootdx_consts' else r.text[:500],flush=True)
 except Exception as e: print(name,str(e),flush=True)

if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=5) as p: list(p.map(probe,targets.items()))
