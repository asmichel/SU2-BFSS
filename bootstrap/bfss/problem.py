"""The exact problem file: the only interface between compiler and solvers.

Numbers in Q(sqrt2) are pairs (a, b) of Fractions meaning a + b sqrt2.  An affine form
in the free coordinates z is (constant, [(k, coefficient), ...]).  Block b is the
Hermitian matrix M_b(z) whose entries (i <= j) are re + i im; the constraints are
M_b(z) >= 0 for all b.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction

SPEC_VERSION = 2                   # 2: observables named as in the paper (u, Y; were V, F)
CONVENTIONS = "T=sigma/sqrt2, tr TT=delta, {theta,theta}=delta, H=T+u+Y"


@dataclass
class Block:
    """One Gram block: its family, row names, and the affine forms of its upper-triangle entries."""
    family: str
    rows: list[str]
    entries: list            # [(i, j, re affine, im affine)]


@dataclass
class Problem:
    level: int
    manifest: dict
    coordinates: list[str]
    blocks: list[Block]
    observables: dict        # {name: affine}
    provenance: dict

    @property
    def dimension(self) -> int:
        return len(self.coordinates)

    def save(self, path) -> None:
        def q2(v):
            return [str(v[0]), str(v[1])]

        def aff(a):
            return {"c": q2(a[0]), "z": [[k, q2(v)] for k, v in a[1]]}

        doc = {"spec_version": SPEC_VERSION, "conventions": CONVENTIONS, "level": self.level,
               "manifest": self.manifest, "provenance": self.provenance, "dimension": self.dimension,
               "coordinates": self.coordinates,
               "blocks": [{"family": b.family, "rows": b.rows, "complex": any(im[1] or any(im[0]) for *_, im in b.entries),
                           "entries": [[i, j, {"re": aff(re), "im": aff(im)}] for i, j, re, im in b.entries]}
                          for b in self.blocks],
               "observables": {k: aff(v) for k, v in self.observables.items()}}
        with open(path, "w") as f:
            json.dump(doc, f, separators=(",", ":"))

    @classmethod
    def load(cls, path) -> "Problem":
        with open(path) as f:
            doc = json.load(f)
        if doc["spec_version"] != SPEC_VERSION:
            raise ValueError(f"problem file has spec version {doc['spec_version']}")

        def q2(v):
            return (Fraction(v[0]), Fraction(v[1]))

        def aff(a):
            return (q2(a["c"]), [(k, q2(v)) for k, v in a["z"]])

        blocks = [Block(b["family"], b["rows"], [(i, j, aff(e["re"]), aff(e["im"])) for i, j, e in b["entries"]])
                  for b in doc["blocks"]]
        return cls(doc["level"], doc["manifest"], doc["coordinates"], blocks,
                   {k: aff(v) for k, v in doc["observables"].items()}, doc["provenance"])
