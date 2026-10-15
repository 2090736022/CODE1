import sys
from concurrent.futures import ThreadPoolExecutor
import requests
sys.stdout.reconfigure(encoding='utf-8')
queries=['股票 历史数据 in:name,description','分钟 数据集 in:name,description','A股 数据 下载 in:name,description','stock minute csv china in:readme','1min A股 in:readme']
def search(q):
 try:
  r=requests.get('https://api.github.com/search/repositories',params={'q':q,'per_page':8},timeout=15)
  print(q,r.status_code,[(x['full_name'],str(x.get('description'))[:150]) for x in r.json().get('items',[])],flush=True)
 except Exception as e: print(q,e)
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=3) as p: list(p.map(search,queries))
