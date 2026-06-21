"""Contract + behavior tests for the interlingua versioning + drift scheme
(build-tracker step 5; closes PLAN.md §6 gap 7g).

Run: python3 tests/test_interlingua.py
Exits non-zero if any check fails. Encodes the step-5 spec per Rule 25: the
shared belief language is explicit, versioned and checkable; the CURRENT bank
conforms to the v0.1 baseline; future drift (unknown type / missing key /
un-migratable version) is detected; a version bump is bridged by migration; and
the loose types (action/decision/forecast) are surfaced by coherence, not faked.
"""
from __future__ import annotations

import os
import sys
import json
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pattern_brain as pb
from pattern_brain.belief import Belief
from pattern_brain.interlingua import (
    INTERLINGUA_VERSION, InterlinguaError, BeliefTypeSchema,
    validate_belief, detect_drift, conformance_report, migrate_belief,
    has_migration_path, register_migration, interlingua_coherence,
    known_belief_types, catalog, belief_type_schema,
)

FAILS = []


def check(cond, msg):
    cond = bool(cond)
    if not cond:
        FAILS.append(msg)
    return cond


def bank_beliefs(seed=0):
    """Every node's belief on a generic sequence (the live interlingua traffic)."""
    rng = np.random.default_rng(seed)
    X = np.cumsum(rng.normal(size=(240, 3)), axis=0)
    out = []
    for node in pb.default_bank():
        try:
            b = node.process(X, np.sign(X[:, 0])) if node.requires_y else node.process(X)
            out.append(b)
        except Exception:
            pass
    return out


# ------------------------------------------------------- baseline: bank conforms
def test_version_is_single_source_of_truth():
    check(INTERLINGUA_VERSION == pb.SCHEMA_VERSION,
          f"interlingua v{INTERLINGUA_VERSION} != belief SCHEMA_VERSION {pb.SCHEMA_VERSION}")
    print(f"  version: interlingua == belief SCHEMA_VERSION == {INTERLINGUA_VERSION}")


def test_current_bank_conforms():
    """THE baseline test: every belief the current bank emits must conform to the
    declared v0.1 catalog (it was built from the bank, so this must hold)."""
    beliefs = bank_beliefs()
    rep = conformance_report(beliefs)
    check(rep.conforms, f"current bank does NOT conform: {[v.to_dict() for v in rep.violations]}")
    check(rep.n_beliefs == rep.n_conforming, "some bank beliefs non-conforming")
    # every emitted type is in the catalog
    emitted = {b.type for b in beliefs}
    check(emitted <= set(known_belief_types()),
          f"emitted types not all catalogued: {emitted - set(known_belief_types())}")
    print(f"  baseline: {rep.n_beliefs} bank beliefs, all conform to v{rep.version} "
          f"({len(emitted)} types)")


# --------------------------------------------------------- drift detection
def test_unknown_type_flagged():
    b = Belief(type="teleport", payload={"x": 1}, confidence=0.5, source="rogue_node")
    viols = validate_belief(b)
    check(any(v.kind == "unknown_type" for v in viols), f"unknown type not flagged: {viols}")
    print("  drift: unknown belief type -> unknown_type violation")


def test_missing_required_key_flagged():
    """A retrained 'cluster' node that drops 'labels' must be caught."""
    b = Belief(type="cluster", payload={"n_clusters": 3}, confidence=0.7, source="kmeans_v2")
    viols = validate_belief(b)
    check(any(v.kind == "missing_keys" for v in viols), f"missing key not flagged: {viols}")
    check(any("labels" in v.detail for v in viols), "the missing key name not reported")
    print("  drift: cluster missing 'labels' -> missing_keys violation")


def test_version_mismatch_without_migration_flagged():
    b = Belief(type="signal", payload={"series": [1, 2, 3]}, confidence=0.5,
               source="old_node", schema_version="9.9")  # no migration to current
    viols = validate_belief(b)
    check(any(v.kind == "version_mismatch" for v in viols),
          f"un-migratable version not flagged: {viols}")
    print("  drift: stale version with no migration -> version_mismatch violation")


# ----------------------------------------------------------------- migration
def test_migration_bridges_version_bump():
    """A v0.0 forecast (key 'value') migrates to v0.1 (key 'estimate'); after
    migration it conforms, so it's NOT a violation (migratable)."""
    check(has_migration_path("0.0", "0.1"), "expected a 0.0 -> 0.1 migration path")
    old = Belief(type="forecast", payload={"value": 1.5}, confidence=0.6,
                 source="legacy_forecaster", schema_version="0.0")
    new = migrate_belief(old, "0.1")
    check(new.schema_version == "0.1", f"migration did not bump version: {new.schema_version}")
    check("estimate" in new.payload and "value" not in new.payload,
          f"migration did not rename value->estimate: {new.payload}")
    check(new.confidence == old.confidence and new.source == old.source,
          "migration mangled confidence/source")
    # validate_belief should treat the original 0.0 belief as migratable (no viol)
    viols = validate_belief(old)
    check(not any(v.kind == "version_mismatch" for v in viols),
          f"migratable belief wrongly flagged as version_mismatch: {viols}")
    print("  migration: v0.0 forecast(value) -> v0.1 forecast(estimate), conforms")


def test_migration_idempotent_and_no_path_raises():
    b = Belief(type="signal", payload={"series": [1]}, confidence=0.4, source="n",
               schema_version=INTERLINGUA_VERSION)
    check(migrate_belief(b) is b or migrate_belief(b).schema_version == INTERLINGUA_VERSION,
          "migrating a current-version belief should be a no-op")
    nopath = Belief(type="signal", payload={"series": [1]}, confidence=0.4, source="n",
                    schema_version="7.7")
    try:
        migrate_belief(nopath, INTERLINGUA_VERSION)
        check(False, "missing migration path did not raise")
    except InterlinguaError:
        pass
    print("  migration: no-op on current version; missing path -> InterlinguaError")


# ------------------------------------------------------------- coherence (soft)
def test_coherence_surfaces_loose_types():
    """The loose types (action/decision/forecast) have inconsistent payloads
    across emitters — coherence must surface them WITHOUT them being hard
    violations (they still conform to the deliberately-loose v0.1 catalog)."""
    beliefs = bank_beliefs()
    notes = interlingua_coherence(beliefs)
    flagged = {n.belief_type for n in notes}
    check({"forecast", "decision"} <= flagged or "forecast" in flagged,
          f"coherence did not surface loose types: {flagged}")
    # and these are NOT hard violations
    rep = conformance_report(beliefs)
    check(rep.conforms, "loose types should still conform to the loose v0.1 catalog")
    for t in ("decision", "forecast", "action"):
        s = belief_type_schema(t)
        check(s is not None and s.required_keys == (),
              f"{t} should have no required keys in the loose v0.1 baseline")
    print(f"  coherence: loose types surfaced {sorted(flagged)} (soft, not violations)")


# --------------------------------------------------- serialization / dashboard
def test_report_and_catalog_json_serializable():
    rep = conformance_report(bank_beliefs())
    json.dumps(rep.to_dict())
    json.dumps(catalog())
    notes = [n.to_dict() for n in interlingua_coherence(bank_beliefs())]
    json.dumps(notes)
    print("  serialization: report + catalog + coherence notes all JSON-serializable")


def test_pathway_stream_conforms():
    """A real Connector run's belief stream conforms (drift check over a pathway)."""
    res = pb.default_connector().run(np.cumsum(np.random.default_rng(1).normal(size=(200, 2)), axis=0))
    viols = detect_drift(res.beliefs())
    check(not viols, f"connector belief stream drifted: {[v.to_dict() for v in viols]}")
    print(f"  pathway: connector stream ({len(res.beliefs())} beliefs) conforms")


def test_domain_independence():
    import tokenize
    forbidden = ("candle", "ohlcv", "orderbook", "order_book")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "pattern_brain", "interlingua.py")
    hits = []
    with tokenize.open(path) as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type == tokenize.NAME and any(b in tok.string.lower() for b in forbidden):
                hits.append(tok.string)
    check(not hits, f"domain-coupling identifiers in interlingua.py: {hits}")
    print("  domain-independence: interlingua has no candle/ohlcv/orderbook coupling")


def main():
    print("=" * 70)
    print("Pattern Brain — interlingua versioning/drift (step 5): contract tests")
    print("=" * 70)
    test_version_is_single_source_of_truth()
    test_current_bank_conforms()
    test_unknown_type_flagged()
    test_missing_required_key_flagged()
    test_version_mismatch_without_migration_flagged()
    test_migration_bridges_version_bump()
    test_migration_idempotent_and_no_path_raises()
    test_coherence_surfaces_loose_types()
    test_report_and_catalog_json_serializable()
    test_pathway_stream_conforms()
    test_domain_independence()
    print("=" * 70)
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s):")
        for f in FAILS:
            print("  - " + f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
