"""Command line interface: import the published data, measure states and optimize ansatzes."""

from __future__ import annotations

import argparse
import json
import logging
import time
from fractions import Fraction
from pathlib import Path

from .ansatz import Ansatz
from .state import Engine, load, save, write_json

REPORTED = ("M", "E", "Haux", "F_lb", "T", "u", "Y", "R2", "R4", "R6", "R8")  # printed and stored by measure


def _basis_of(path: Path):
    """The basis of an ansatz or state file at its canonical widths (largest longitudinal width 1)."""
    data = json.loads(path.read_text())
    basis = Ansatz.from_json(data).basis() if data.get("schema") == "bfss.ansatz.v1" else load(path).basis
    return basis.dilated(1 / basis.reference_width())


def _ball(v) -> str:
    """Midpoint and rigorous radius of a ball."""
    return f"{v.mid().str(20, radius=False)} +/- {v.rad().str(2, radius=False)}"


def _store(state, values: dict, path: Path):
    """Save a state with the midpoints of its measured observables in the metadata."""
    state.metadata["observables"] = {k: values[k].mid().str(20, radius=False) if k != "F_lb" else repr(values[k])
                                     for k in REPORTED}
    save(state, path)


def _report(values: dict):
    for key in REPORTED:
        v = values[key]
        print(f"{key:>4} = {v if isinstance(v, float) else _ball(v)}")
    print("canonical blocks (lambda = 1):")
    for key, v in values["canonical_blocks"].items():
        print(f"  {key:>6} = {_ball(v)}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bfss", description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".cache"), help="operator and contraction cache")
    parser.add_argument("--workers", type=int, help="worker processes (default: from CPUs and available memory)")
    parser.add_argument("--prec", type=int, default=256, help="bits of ball arithmetic for matrix elements and frames")
    parser.add_argument("--exact", action="store_true", help="exact rational matrix elements (slow; for validation)")
    parser.add_argument("--threads", type=int, help="threads for dense ball/rational products (default: all CPUs)")
    parser.add_argument("--verbose", action="store_true", help="log the stages of the computation")
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import-reference", help="convert the published companion data into a state file")
    imp.add_argument("source", type=Path, help="a directory holding the published companion files")
    imp.add_argument("-o", "--output", type=Path, required=True)
    mea = sub.add_parser("measure", help="norm, E, <H_aux> and its bound on F, <T>, <u>, <Y> and radii of a state")
    mea.add_argument("state", type=Path)
    mea.add_argument("--store", action="store_true", help="also store the observables in the state file")
    opt = sub.add_parser("optimize", help="variational coefficients and dilation for the basis of a state or ansatz")
    opt.add_argument("input", type=Path)
    opt.add_argument("-o", "--output", type=Path, required=True)
    opt.add_argument("--dilation", type=Fraction, help="fixed instead of optimized (a decimal or fraction)")
    con = sub.add_parser("contract", help="compute and cache the contraction table of an ansatz or state")
    con.add_argument("input", type=Path)
    ans = sub.add_parser("ansatz", help="write the shell specification of a state's basis")
    ans.add_argument("state", type=Path)
    ans.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.verbose:
        logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

    engine = Engine(args.cache, args.workers, args.prec, args.exact, args.threads)
    start = time.time()
    if args.command == "import-reference":
        from .reference import reference_state
        save(reference_state(args.source), args.output)
    elif args.command == "measure":
        state = load(args.state)
        values = engine.measure(state)
        if args.store:
            _store(state, values, args.state)
        _report(values)
    elif args.command == "optimize":
        state = engine.optimize(_basis_of(args.input), dilation=args.dilation)
        save(state, args.output)  # the optimum first; the measured observables are added below
        values = engine.measure(state)
        _store(state, values, args.output)
        _report(values)
    elif args.command == "contract":
        engine.table(_basis_of(args.input))
    elif args.command == "ansatz":
        write_json(args.output, Ansatz.of(_basis_of(args.state)).to_json())
    print(f"[{args.command} finished in {time.time() - start:.1f}s]")


if __name__ == "__main__":
    main()
