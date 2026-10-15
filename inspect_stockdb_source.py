import requests, sys
sys.stdout.reconfigure(encoding='utf-8')
base='https://raw.githubusercontent.com/hello245m/free-stockdb/main/'
for path in ['docs/DATA_SOURCE.md']:
 r=requests.get(base+path,timeout=20)
 print(path,r.status_code,r.text[:12000],flush=True)
r=requests.get('https://api.github.com/repos/hello245m/free-stockdb/git/trees/main',params={'recursive':'1'},timeout=20)
print('TREE',[(x['path'],x.get('size')) for x in r.json().get('tree',[]) if 'sync' in x['path']],flush=True)
r=requests.get('https://api.github.com/repos/hello245m/free-stockdb/releases',timeout=20)
print('RELEASES',[(v['tag_name'],[(a['name'],a['size'],a['browser_download_url']) for a in v['assets']]) for v in r.json()[:3]],flush=True)
r=requests.get('https://api.github.com/repos/iwerevip/free-stockdb_1/git/trees/main',params={'recursive':'1'},timeout=20)
print('OLD TREE',[(x['path'],x.get('size')) for x in r.json().get('tree',[]) if any(s in x['path'] for s in ['sync','更新','data','csv','zip'])][:50],flush=True)
