import sys, collections, joblib, numpy as np
for p in sys.argv[1:]:
    d = joblib.load(p)
    seqs = d if isinstance(d, list) else d.get("tokens", d.get("data", d)) if isinstance(d, dict) else d
    if isinstance(d, dict):
        print("keys", list(d.keys())[:10])
    L = np.array([len(x["tokens"] if isinstance(x, dict) else x) for x in seqs])
    print(p.split("/")[-1][:70], "n", len(L), "mean %.1f p99 %d max %d" % (L.mean(), np.percentile(L, 99), L.max()),
          "n>111:", int((L > 111).sum()), "n>=192:", int((L >= 192).sum()))
