import io, zipfile, requests, sys, re
sys.stdout.reconfigure(encoding='utf-8')
r=requests.get('https://api.github.com/repos/hello245m/free-stockdb/releases',timeout=20)
asset=next(a for a in r.json()[0]['assets'] if 'windows' in a['name'] and a['name'].endswith('.zip'))
r=requests.get(asset['browser_download_url'],timeout=30); r.raise_for_status()
z=zipfile.ZipFile(io.BytesIO(r.content))
print([(i.filename,i.file_size) for i in z.infolist()],flush=True)
for i in z.infolist():
 if i.filename.endswith('数据更新.exe'):
  strings=re.findall(rb'[\x20-\x7e]{6,}',z.read(i))
  for n,s in enumerate(strings):
   if s==b'"files"' or b'X-Sync-UA' in s:
    print('\n'.join(x.decode() for x in strings[max(0,n-35):n+45]),flush=True)
