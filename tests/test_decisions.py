"""Tests for the resolved §7 open decisions (final follow-ons).

Run: python3 tests/test_decisions.py
Covers: the genome manifest (the resolved "cpu ml models" deliverable), the
defined personas for every agent (Rule 17/19), and that each evolution engine
carries its persona. Domain-agnostic (Rule 23).
"""
from __future__ import annotations

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pattern_brain as pb

FAILS = []


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def test_genome_manifest():
    """The deliverable = a manifest of the whole bank + build requirements, not
    committed weights."""
    m = pb.genome_manifest()
    check(m["n_nodes"] == len(m["nodes"]) >= 55, f"manifest node count off: {m['n_nodes']}")
    check(set(m["by_layer"]) == set(m["layers"]), "by_layer/layers mismatch")
    check(sum(len(v) for v in m["by_layer"].values()) == m["n_nodes"],
          "by_layer does not partition all nodes")
    check("light" in m["requirements"] and "deep_optional" in m["requirements"],
          "manifest missing build requirements")
    check(all("node_type" in n and "layer" in n for n in m["nodes"]),
          "manifest node entries missing metadata")
    json.dumps(m)  # must serialize (it IS the deliverable artifact)
    print(f"  manifest: {m['n_nodes']} nodes across {len(m['layers'])} layers, serializes")


def test_personas_defined():
    p = pb.all_personas()
    for key in ("connector", "feature1_model", "feature3_pathway", "feature2_algorithm"):
        check(key in p, f"missing persona {key}")
        check(p[key]["name"] and p[key]["role"] and p[key]["voice"], f"{key} persona incomplete")
    names = {p[k]["name"] for k in p}
    check("The Synthesist" in names, "Connector persona (The Synthesist) missing")
    check(len(names) == len(p), "personas should have distinct names")
    print(f"  personas: {sorted(names)}")


def test_engines_carry_persona():
    """Each evolution engine exposes its defined role-persona (Rule 17)."""
    pairs = [(pb.ModelEvolver, "The Breeder"), (pb.PathwayEvolver, "The Architect"),
             (pb.AlgorithmEvolver, "The Inventor")]
    for Eng, expected in pairs:
        per = Eng().persona
        check(per.name == expected, f"{Eng.__name__} persona {per.name!r} != {expected!r}")
    print("  engines: Model->Breeder, Pathway->Architect, Algorithm->Inventor")


def test_step10_slots_are_function_level():
    """§7 Step-10 resolution: the functional LAYERS are the slots Feature 3
    connects (function-level), with any algorithm interchangeable as the filler.
    Verify the bank really is organized by function layer with multiple algorithms
    per slot."""
    by_layer = {l: pb.by_layer(l) for l in pb.layers()}
    multi = [l for l, members in by_layer.items() if len(members) >= 2]
    check(len(multi) >= 6, f"most layers should hold several interchangeable algos: {multi}")
    print(f"  step10: {len(by_layer)} function-level slots; {len(multi)} hold >=2 interchangeable algos")


def main():
    print("=" * 70)
    print("Pattern Brain — resolved §7 decisions: manifest, personas, step-10")
    print("=" * 70)
    test_genome_manifest()
    test_personas_defined()
    test_engines_carry_persona()
    test_step10_slots_are_function_level()
    print("=" * 70)
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s):")
        for f in FAILS:
            print("  - " + f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
