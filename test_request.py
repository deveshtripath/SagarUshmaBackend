"""With the server running:  python test_request.py   (sends random grids to /api/predict/)"""
import json
import random
import urllib.request

T, H, W = 3, 64, 64
CH = {"sst": 3, "sss": 1, "ssh": 1, "current": 3, "wind": 3}


def grid(c):
    return [[[[random.gauss(0, 1) for _ in range(W)] for _ in range(H)] for _ in range(c)] for _ in range(T)]


body = {"depth": 100, "inputs": {k: grid(c) for k, c in CH.items()}}
req = urllib.request.Request(
    "http://127.0.0.1:8000/api/predict/",
    json.dumps(body).encode(),
    {"Content-Type": "application/json"},
)
out = json.load(urllib.request.urlopen(req))
print("shape:", out["shape"], "channels:", list(out["channels"]))
