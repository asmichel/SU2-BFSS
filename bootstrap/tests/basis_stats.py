"""Rows of the complete basis at a level, per irrep: multiplicities, tail-filtered rows, sizes.

  python tests/basis_stats.py LEVEL [WORKERS]
"""
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bfss import basis as B

LEVEL = int(sys.argv[1])
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 1


def work(item):
    (g, lam), m = item
    t = time.time()
    s = B.sector(g, lam, m)
    return g, lam, m, len(s.mons), len(s.rows), s.tails, [sum(z != B.Z0 for z in v) for v in s.rows], round(time.time() - t, 1)


if __name__ == "__main__":
    B.action()
    mult = B.multiplicities(LEVEL)
    items = sorted(mult.items(), key=lambda kv: -B.grade_level(kv[0][0]))
    pool = mp.get_context("fork").Pool(WORKERS)
    per = defaultdict(list)
    t0 = time.time()
    for g, lam, m, nm, nr, tails, sizes, sec in pool.imap_unordered(work, items):
        per[lam].append((g, m, nr, tails, sizes))
        print(json.dumps({"grade": g, "lam": lam, "mons": nm, "hw": m, "rows": nr, "tails": tails, "terms": sizes, "s": sec}), flush=True)
    print(f"level {LEVEL}: {sum(mult.values())} multiplets, {sum(r for v in per.values() for _, _, r, _, _ in v)} rows,"
          f" {len(per)} irreps, {time.time() - t0:.0f} s")
    for lam, v in sorted(per.items(), key=lambda kv: -sum(r for _, _, r, _, _ in kv[1])):
        rows = sum(r for _, _, r, _, _ in v)
        if rows:
            terms = sum(sum(s) for *_, s in v)
            print(f"  {lam} dim {B.dimension(lam):5d} rows {rows:4d} hw-terms {terms:7d}  dim*terms {B.dimension(lam) * terms:9d}")
