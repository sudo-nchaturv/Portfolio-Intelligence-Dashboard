import requests
import json
import os

token = ''
with open('.streamlit/secrets.toml', 'r') as f:
    for line in f:
        if 'INDSTOCKS_TOKEN' in line:
            token = line.split('=')[1].strip().replace("'", "").replace('"', '')
            break

def check(end_ms):
    start_ms = end_ms - (365 * 86400000)
    r = requests.get(
        'https://api.indstocks.com/market/historical/1day',
        headers={'Authorization': token},
        params={'scrip-codes': '4992', 'start_time': start_ms, 'end_time': end_ms}
    )
    print(f'End MS {end_ms} -> {r.status_code}')
    if r.status_code == 200:
        print(f'Candles: {len(r.json().get("data", {}).get("candles", []))}')

check(1711019124000)
check(1735689600000)
