"""Command line: compile a hierarchy into an exact problem file, solve, scan, certify, plot.

  python run.py compile --level 10 --workers 32 --out runs/L10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

from bfss import moments, reduction, sdp
from bfss.problem import Problem


def compile_run(level: int, out: Path, workers: int, only: list | None = None) -> None:
    """Compile the hierarchy of a level into out/problem.json, writing the manifest and a log alongside."""
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "compile.log", "a")

    def log(line: str) -> None:
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(json.dumps({"event": "start", "level": level, "workers": workers, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    h = moments.canonical(level, workers, log)
    (out / "manifest.json").write_text(json.dumps(h.manifest(), indent=1))
    log(json.dumps({"event": "hierarchy", "blocks": len(h.rows), "rows": sum(len(r) for r in h.rows.values()),
                    "ward_forms": len(h.ward)}))
    prob = reduction.compile_hierarchy(h, workers=workers, log=log, cache=out / "projections", only=only)
    if prob is None:
        return
    prob.save(out / "problem.json")
    log(json.dumps({"event": "done", "dimension": prob.dimension, **{k: prob.provenance[k] for k in ("rank", "ward_rank", "seconds")}}))


def cmd_compile(a) -> None:
    """The hierarchy of a level and its exact problem in a run directory."""
    compile_run(a.level, Path(a.out), a.workers, [int(t) for t in a.primes.split(",")] if a.primes else None)


def _parse_obj(text: str) -> dict:
    """'R2' or 'u+0.5*R2' or '-u' -> {observable: weight}."""
    out = {}
    for part in filter(None, text.replace("-", "+-").split("+")):
        w, _, name = part.rpartition("*")
        if not w and name.startswith("-"):
            w, name = "-1", name[1:]
        out[name] = Fraction(w or 1)
    return out


def cmd_solve(a) -> None:
    run = Path(a.run)
    prob = Problem.load(run / "problem.json")
    fixed = dict(kv.split("=") for kv in a.fix) if a.fix else None
    S = sdp.specialize(prob, _parse_obj(a.min), fixed)
    if a.strict:
        S = sdp.strictness(S)
    if a.backend == "clarabel":
        res = sdp.solve_clarabel(S)
    else:
        res = sdp.solve_sdpb(S, run / "sdpb" / a.name, precision=a.precision, procs=a.procs, threshold=a.threshold,
                             max_iter=a.max_iter)
    res.update(objective=a.min, fixed=a.fix, strict=a.strict)
    (run / "solves").mkdir(exist_ok=True)
    (run / "solves" / f"{a.name}.{a.backend}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("backend", "status", "value", "primal", "dual", "verified")}))
    print("min eigs:", res["min_eigs"])


def cmd_certify(a) -> None:
    """Exact certificate of a saved SDPB solve (same objective and constraints as when it was solved)."""
    from bfss import certify
    run = Path(a.run)
    res = json.loads((run / "solves" / f"{a.name}.sdpb.json").read_text())
    fixed = dict(kv.split("=") for kv in res["fixed"]) if res.get("fixed") else None
    S = sdp.specialize(Problem.load(run / "problem.json"), _parse_obj(res["objective"]), fixed)
    out = certify.certify(S, run / "sdpb" / a.name / "out")
    sys.set_int_max_str_digits(0)
    doc = {"name": a.name, "beta": [str(out["beta"][0]), str(out["beta"][1])], "residual_l1": out["residual_l1"],
           "bound": out["bound"], "Z": out["Z"], "min_pivot": out["min_pivot"], "sdpb_value": res["value"]}
    (run / "solves" / f"{a.name}.certificate.json").write_text(json.dumps(doc))
    print(json.dumps({k: v for k, v in doc.items() if k != "beta"} | {"beta": out["beta_digits"]}))


def cmd_scan(a) -> None:
    from bfss import regions
    regions.TOLERANCE = a.tolerance
    run = Path(a.run)
    lower = [s for r in a.lower for s in regions.load_solves(Path(r))]      # bounds of lower levels hold here
    res = regions.scan(Problem.load(run / "problem.json"), run, a.planes, a.n, a.backend, a.workers, a.precision,
                       a.threshold, lower=lower, procs=a.procs, max_iter=a.max_iter)
    doc = regions.write(run, res, lower)
    for pl, g in doc["regions"].items():
        print(json.dumps({k: g.get(k) for k in ("plane", "bounds", "inherited", "points", "gap", "violation", "ok")}),
              flush=True)


def cmd_plot(a) -> None:
    from bfss import plots
    plots.figure([Path(r) for r in a.runs], Path(a.out), tag=a.tag)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile", help="hierarchy at level L -> problem.json")
    c.add_argument("--level", type=int, required=True)
    c.add_argument("--workers", type=int, default=1)
    c.add_argument("--out", required=True)
    c.add_argument("--primes", help="only project these prime indices into the cache (e.g. 1,2), for parallel jobs")
    s = sub.add_parser("solve", help="minimize an observable combination on a compiled problem")
    s.add_argument("run")
    s.add_argument("--min", required=True, help="e.g. R2, -u, u+0.5*R2")
    s.add_argument("--fix", nargs="*", help="exact constraints, e.g. R2=806026/100000")
    s.add_argument("--strict", action="store_true", help="strict-feasibility diagnostic instead")
    s.add_argument("--backend", choices=("clarabel", "sdpb"), default="clarabel")
    s.add_argument("--precision", type=int, default=1024)
    s.add_argument("--procs", type=int, default=1)
    s.add_argument("--threshold", type=float, default=1e-30)
    s.add_argument("--name", default="tip")
    s.add_argument("--max-iter", type=int, default=1000)
    r = sub.add_parser("scan", help="support scans in observable planes -> regions.json")
    r.add_argument("run")
    r.add_argument("--planes", nargs="+", default=["R2_u", "R2_R4", "R4_u"])
    r.add_argument("--n", type=int, default=96)
    r.add_argument("--tolerance", type=float, default=1e-3, help="accuracy of the regions (window units)")
    r.add_argument("--procs", type=int, default=1, help="SDPB processes per solve")
    r.add_argument("--max-iter", type=int, default=1000, help="SDPB iterations per solve (bounds stay certified)")
    r.add_argument("--backend", choices=("clarabel", "sdpb"), default="sdpb")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--precision", type=int, default=1024)
    r.add_argument("--threshold", type=float, default=1e-30)
    r.add_argument("--lower", nargs="*", default=[], help="run directories of lower levels of the same hierarchy")
    e = sub.add_parser("certify", help="exact certificate of a saved SDPB solve")
    e.add_argument("run")
    e.add_argument("--name", default="tip")
    f = sub.add_parser("plot", help="paper figure from regions.json of one or more runs")
    f.add_argument("runs", nargs="+")
    f.add_argument("--out", required=True)
    f.add_argument("--tag", default="peninsula", help="file name prefix (figures/ uses regions)")
    a = ap.parse_args(argv)
    {"compile": cmd_compile, "solve": cmd_solve, "scan": cmd_scan, "certify": cmd_certify, "plot": cmd_plot}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
