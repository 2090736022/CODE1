import requests,sys
import importlib.util
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / '.vendor' / 'msgpack'))
from concurrent.futures import ThreadPoolExecutor
sys.stdout.reconfigure(encoding='utf-8')
urls=['https://ah.123128.xyz/.sync_manifest.json','https://ad.123128.xyz/.sync_manifest.json','http://8.138.149.215:7899/?cmd=get&t=分钟k:000001:20260818094500']
def probe(url):
 try:
  r=requests.get(url,timeout=15)
  if '7899' in url:
   import msgpack
   print(url,r.status_code,r.headers.get('Content-Type'),msgpack.unpackb(r.content,raw=False),flush=True)
  else: print(url,r.status_code,len(r.content),r.text[:1000],flush=True)
 except Exception as e: print(url,e,flush=True)
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=4) as p:list(p.map(probe,urls))
