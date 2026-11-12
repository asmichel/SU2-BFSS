"""The measurement, tail-fit and EFT density records in results/ describe the committed states they name."""

import hashlib
import json
import sys
from decimal import Decimal, getcontext
from fractions import Fraction

import numpy as np
import pytest
from conftest import ROOT

sys.path.insert(0, str(ROOT / "analysis"))
import assemble_eft_density  # noqa: E402
import curate_tail_fit  # noqa: E402
import eft_density  # noqa: E402

RESULTS = ROOT / "results"
MEASUREMENTS = sorted(RESULTS.glob("*_measurement.json"))
TAILS = sorted(RESULTS.glob("*_tail_fit.json"))
DENSITIES = sorted(RESULTS.glob("*_eft_density.json"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("path", MEASUREMENTS, ids=lambda p: p.name)
def test_measurement_records_derive_from_their_states(path):
    """Hash of the named state; F_lb = 1 - kappa <H_aux> (key Haux) and T = dilation x canonical T from the printed
    midpoints; the 256-bit values agree with the observables the optimizer stored in the state."""
    getcontext().prec = 80
    record = json.loads(path.read_text())
    state_path = ROOT / record["state_file"]
    assert record["state_sha256"] == sha256(state_path)
    obs = record["observables"]
    kappa = (Decimal(36) + 8 * Decimal(14).sqrt()) / 25
    assert str(1 - kappa * Decimal(obs["Haux"])).startswith(obs["F_lb"])
    lam = Fraction(json.loads(state_path.read_text())["metadata"]["dilation"])
    assert Decimal(obs["T"]) == Decimal(lam.numerator) / Decimal(lam.denominator) * Decimal(record["canonical_blocks"]["T"])
    stored = json.loads(state_path.read_text())["metadata"]["observables"]
    for key in ("M", "E", "Haux", "R2", "R4"):
        assert float(obs[key]) == pytest.approx(float(stored[key]), rel=1e-12), key


@pytest.mark.parametrize("path", TAILS, ids=lambda p: p.name)
def test_tail_records_derive_from_their_profiles(path):
    """Hash of the named state; the reported window is the flattest of ratio 1.1; the local-power minimum and
    its 2% band follow from the recorded density as curate_tail_fit derives them (the extended record, which
    predates the script, locates the minimum to 1e-8)."""
    record = json.loads(path.read_text())
    assert record["state_sha256"] == sha256(ROOT / record["state_file"])
    reported, flattest = record["reported"], record["window_variations"]["flattest_by_ratio"]["1.1"]
    assert all(reported[k] == flattest[k] for k in flattest)
    rmin, pmin, lo, hi = curate_tail_fit.local_power_band(np.array(record["profile"]["r"]),
                                                          np.array(record["profile"]["p"]))
    assert reported["local_power_minimum"]["value"] == pytest.approx(pmin, rel=1e-12)
    assert reported["local_power_minimum"]["r"] == pytest.approx(rmin, abs=1e-7)
    assert reported["local_power_within_2_percent_of_10"] == pytest.approx([lo, hi], rel=1e-12)


def assert_close(recorded, expected, path="validation"):
    """Nested records agree: equal strings and structure, numbers to rounding."""
    if isinstance(expected, dict):
        assert set(recorded) == set(expected), path
        for key in expected:
            assert_close(recorded[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(recorded) == len(expected), path
        for i, (a, b) in enumerate(zip(recorded, expected)):
            assert_close(a, b, f"{path}[{i}]")
    elif isinstance(expected, str) or expected is None:
        assert recorded == expected, path
    else:
        assert recorded == pytest.approx(expected, rel=1e-12, abs=1e-300), path


@pytest.mark.parametrize("path", DENSITIES, ids=lambda p: p.name)
def test_eft_density_records_derive_from_their_sections(path):
    """Hashes of the named state and tail record; each section's top-level values are its highest order, its
    order agreement follows from by_order, rho2_over_eft, rho4_over_rho2_sq and p_cond are the moments they are
    defined from, and the validation residuals follow from the sections and the tail profile as the assembler
    computes them."""
    record = json.loads(path.read_text())
    assert record["state_sha256"] == sha256(ROOT / record["state"])
    assert record["sources"]["tail_sha256"] == sha256(ROOT / record["sources"]["tail"])
    for name in ("radial", "joint", "transverse", "conditional"):
        section = record[name]
        top = section["by_order"][str(section["orders"][-1])]
        assert all(section[key] == value for key, value in top.items()), name
        assert_close(section["order_agreement"], eft_density.agreement(section["by_order"], section["orders"]))
        for values in section["by_order"].values():
            if "rho2_over_eft" in values:
                r, rho2, rho4 = (np.asarray(v) for v in (section["r"], values["rho2"], values["rho4"]))
                assert values["rho2_over_eft"] == pytest.approx(r * rho2 / (4 * np.sqrt(2)), rel=1e-14)
                assert values["rho4_over_rho2_sq"] == pytest.approx(rho4 / rho2**2, rel=1e-14)
            if "p_cond" in values:
                assert values["p_cond"] == pytest.approx(np.asarray(values["p"]) / values["p_radial"], rel=1e-14)
    tail = assemble_eft_density.read_tail(ROOT / record["sources"]["tail"], record)
    expected = assemble_eft_density.validation(record, tail)
    assert_close({key: record["validation"][key] for key in expected}, expected)


@pytest.mark.slow
def test_measurement_record_is_reproduced(engine):
    """bfss measure of the published-basis optimum reproduces its 256-bit measurement record: every printed
    midpoint to its 20 digits, every ball radius to 10% (threaded products can change its last printed digit)."""
    from bfss.state import load

    record = json.loads((RESULTS / "optimized_measurement.json").read_text())
    assert record["precision_bits"] == engine.prec
    measured = engine.measure(load(ROOT / record["state_file"]))
    for key, radius in record["ball_radii"].items():  # the printed balls; the derived entries have no radius
        group, name = key.split(".") if "." in key else ("observables", key)
        ball = measured["canonical_blocks"][name] if group == "canonical" else measured[name]
        printed = record["canonical_blocks" if group == "canonical" else "observables"][name]
        assert ball.mid().str(20, radius=False) == printed, key
        assert float(ball.rad().str(2, radius=False)) == pytest.approx(float(radius), rel=0.1), key
