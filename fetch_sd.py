"""Fetch only the 716 San Diego columns of the LargeST 5-min flow files, without downloading the 7 GB per year.

The LargeST .h5 files are pandas 'fixed' stores: /t/block0_values is one contiguous, uncompressed float64 array
of shape (T, 8600), rows = timestamps, columns in ca_meta.csv order. The SD sensors (District 11) occupy a
contiguous block of 716 columns, so we read row chunks with parallel HTTP range requests and keep that block.
Output: data/sd_his_5min_{year}.npz with values (T, 716), times (T,) datetime64[ns], ids (716,).
Equivalent to `pd.read_hdf(f)[sd_ids]` on the full file (checked on the first rows below).
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
import h5py
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data"
BASE = "https://huggingface.co/datasets/ttydsc30/TrafficForecast-LargeST-Project-Dataset/resolve/main/data/raw"
ROWS_PER_REQ, WORKERS = 96, 24

meta = pd.read_csv(RAW / "ca_meta.csv")
sd_rows = meta.index[meta["District"] == 11].to_numpy()
c0, nsd = int(sd_rows[0]), len(sd_rows)
assert nsd == 716 and (np.diff(sd_rows) == 1).all(), "SD columns are expected to be one contiguous block"

for year in map(int, sys.argv[1:] or ["2020", "2021"]):
    out = RAW / f"sd_his_5min_{year}.npz"
    if out.exists():
        print(out.name, "exists"); continue
    url = f"{BASE}/{year}.h5"
    h = h5py.File(fsspec.open(url, "rb", block_size=1 << 20).open(), "r")
    ds = h["t/block0_values"]
    T, NC = ds.shape
    off = ds.id.get_offset()
    items = h["t/block0_items"][:].astype(str)
    assert (items[c0:c0 + nsd] == meta["ID"].astype(str).to_numpy()[sd_rows]).all()
    times = pd.to_datetime(h["t/axis1"][:])          # stored as int64 ns
    head = ds[:4, c0:c0 + nsd]                          # reference values read through h5py
    real = requests.head(url, allow_redirects=True).url   # resolved CDN URL, supports Range
    vals = np.empty((T, nsd), dtype=np.float64)

    def get(r0):
        r1 = min(r0 + ROWS_PER_REQ, T)
        a, b = off + r0 * NC * 8, off + r1 * NC * 8 - 1
        for _ in range(5):
            try:
                resp = requests.get(real, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
                if resp.status_code == 206 and len(resp.content) == b - a + 1:
                    block = np.frombuffer(resp.content, dtype="<f8").reshape(r1 - r0, NC)
                    vals[r0:r1] = block[:, c0:c0 + nsd]
                    return r1 - r0
            except requests.RequestException:
                pass
        raise RuntimeError(f"failed rows {r0}-{r1}")

    done = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        for n in ex.map(get, range(0, T, ROWS_PER_REQ)):
            done += n
            if done % (ROWS_PER_REQ * 100) < ROWS_PER_REQ:
                print(f"{year}: {done / T:.0%}", flush=True)
    np.testing.assert_array_equal(vals[:4], head)
    np.savez(out, values=vals, times=times.values, ids=items[c0:c0 + nsd])
    print(year, vals.shape, times.min(), times.max(), "saved", flush=True)
