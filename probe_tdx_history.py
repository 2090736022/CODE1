"""Read-only checks of public TDX one-minute history retention."""
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.append(str(Path(__file__).resolve().parent / '.vendor' / 'pytdx'))
from pytdx.hq import TdxHq_API
from pytdx.config.hosts import hq_hosts


def probe(server):
    _, host, port = server
    api = TdxHq_API(raise_exception=True)
    try:
        if not api.connect(host, port, time_out=2):
            return
        rows = api.get_security_bars(8, 0, '000001', 5600, 800)
        print(host, port, len(rows or []), rows[0] if rows else None, rows[-1] if rows else None, flush=True)
    except Exception as error:
        print(host, type(error).__name__, str(error)[:80], flush=True)
    finally:
        try:
            api.disconnect()
        except Exception:
            pass


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=12) as pool:
        hosts = ['110.41.147.114','8.129.13.54','120.24.149.49','47.113.94.204','8.129.174.169','110.41.154.219','124.70.176.52','47.100.236.28','101.133.214.242','47.116.21.80','47.116.105.28','124.70.199.56','121.36.54.217','121.36.81.195','123.249.15.60','124.71.85.110','139.9.51.18','139.159.239.163','106.14.201.131','106.14.190.242','121.36.225.169','123.60.70.228','123.60.73.44','124.70.133.119','124.71.187.72','124.71.187.122','119.97.185.59','47.107.64.168','124.70.75.113','124.71.9.153']
        list(pool.map(probe, [('public', host, 7709) for host in hosts]))
