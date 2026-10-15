import sys
from concurrent.futures import ThreadPoolExecutor
import requests
sys.stdout.reconfigure(encoding='utf-8')
repos=['hello245m/free-stockdb','iwerevip/free-stockdb_1','xbfighting/tdx2db','luke0708/stock-data']
def inspect(repo):
 for branch in ['main','master']:
  r=requests.get(f'https://raw.githubusercontent.com/{repo}/{branch}/README.md',timeout=20)
  if r.status_code==200:
   print('\nREPO',repo,'BRANCH',branch,'\n',r.text[:19000],flush=True)
   return
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=4) as p: list(p.map(inspect,repos))
