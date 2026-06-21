"""The interlingua versioning + drift scheme (build-tracker step 5; closes
PLAN.md §6 gap 7g — "interlingua versioning/drift as models get added/retrained").

The "interlingua" is the Universal Belief Space (``Belief``, Block 5) — the shared
language every node speaks. As the bank grows and nodes get retrained/replaced,
their emitted beliefs can silently drift from that shared contract (a new payload
key, a renamed field, a missing field, an unknown belief type, a bumped version).
This module makes the contract **explicit, versioned, and checkable**:

* a **catalog** — the declared schema of every belief ``type`` at the current
  interlingua version (which payload keys are required vs optional). The v0.1
  catalog is built from what the current bank actually emits, so the existing
  bank conforms by construction — it is the *baseline* future drift is measured
  against.
* **conformance / drift detection** — ``validate_belief`` / ``detect_drift`` /
  ``conformance_report`` flag beliefs that fall off the contract (unknown type,
  missing required keys, an un-migratable version mismatch).
* **migration** — ``register_migration`` / ``migrate_belief`` upgrade an
  older-versioned belief to the current interlingua, so old and new nodes can
  still interoperate across a version bump instead of breaking.
* **coherence** — ``interlingua_coherence`` surfaces a *softer* problem the
  evidence revealed: belief types (action/decision/forecast) whose payloads are
  inconsistent across the nodes that emit them. Not a hard violation, but the
  early-warning sign of an interlingua that needs tightening at the next version.

Single source of truth: ``INTERLINGUA_VERSION`` IS ``belief.SCHEMA_VERSION`` —
the version stamped on every belief is the interlingua version it claims to speak.

Domain-agnostic (PLAN.md §0 / Rule 23): this reasons only about generic belief
types and payload keys. Nothing here references candles/order books.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .belief import Belief, SCHEMA_VERSION

# The interlingua version == the version every Belief stamps itself with.
INTERLINGUA_VERSION: str = SCHEMA_VERSION


class InterlinguaError(RuntimeError):
    """Raised when a belief cannot be migrated to the requested version."""


# --------------------------------------------------------------- the catalog
@dataclass(frozen=True)
class BeliefTypeSchema:
    """The declared contract for one belief ``type`` at the interlingua version."""
    type: str
    required_keys: Tuple[str, ...] = ()
    optional_keys: Tuple[str, ...] = ()
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "required_keys": list(self.required_keys),
            "optional_keys": list(self.optional_keys),
            "description": self.description,
        }


_CATALOG: Dict[str, BeliefTypeSchema] = {}


def register_belief_type(schema: BeliefTypeSchema) -> BeliefTypeSchema:
    """Add/replace a belief-type schema in the current interlingua catalog."""
    _CATALOG[schema.type] = schema
    return schema


def belief_type_schema(belief_type: str) -> Optional[BeliefTypeSchema]:
    return _CATALOG.get(belief_type)


def known_belief_types() -> List[str]:
    return sorted(_CATALOG)


def catalog() -> Dict[str, Dict[str, Any]]:
    """The whole declared interlingua, JSON-serializable (dashboard/audit)."""
    return {t: s.to_dict() for t, s in sorted(_CATALOG.items())}


# The v0.1 interlingua, declared from what the starter bank actually emits so the
# current bank conforms (the honest baseline). Types whose payloads vary across
# their emitters (action/decision/forecast) carry NO required key on purpose — the
# inconsistency is reported separately by interlingua_coherence(), not faked away.
def _seed_catalog_v0_1() -> None:
    _CATALOG.clear()
    R = register_belief_type
    R(BeliefTypeSchema("signal", ("series",),
                       ("direction", "mean_abs_change", "mean_envelope", "slope"),
                       "A derived/cleaned generic sequence."))
    R(BeliefTypeSchema("spectral", ("dominant_freq", "energy", "spectrum"), (),
                       "Frequency-domain description of the sequence."))
    R(BeliefTypeSchema("denoised", ("residual", "series"), ("explained_var", "snr"),
                       "A noise-reduced sequence plus its residual."))
    R(BeliefTypeSchema("anomaly", ("flags", "fraction", "n_anomalies", "scores"), (),
                       "Per-step anomaly flags/scores."))
    R(BeliefTypeSchema("cluster", ("labels", "n_clusters"), ("centers", "n_noise"),
                       "A partition of the sequence into clusters."))
    R(BeliefTypeSchema("density", ("bic", "labels", "max_proba"), (),
                       "A probabilistic mixture assignment."))
    R(BeliefTypeSchema("forecast", (),
                       ("estimate", "next_vector", "order", "std", "variance"),
                       "A prediction of the next value(s). LOOSE: payload varies "
                       "across forecasters — candidate for tightening (see coherence)."))
    R(BeliefTypeSchema("regime", ("current_state", "log_likelihood",
                                  "next_distribution", "next_state", "state_posterior"),
                       (), "A latent-state/regime belief (HMM-family)."))
    R(BeliefTypeSchema("next_state", ("current_state", "distribution", "next_state"),
                       (), "A discrete next-state distribution (Markov-family)."))
    R(BeliefTypeSchema("equation", ("coef", "r2"),
                       ("bic", "degree", "expression", "form", "intercept"),
                       "A fitted equation/relationship."))
    R(BeliefTypeSchema("decision", (),
                       ("action", "max_proba", "predictions", "vote", "z"),
                       "A discrete decision. LOOSE: payload varies across deciders "
                       "— candidate for requiring 'action' next version (coherence)."))
    R(BeliefTypeSchema("action", (),
                       ("action", "arm", "policy", "q_values", "state", "ucb", "values"),
                       "An RL action/policy output. LOOSE: payload varies across "
                       "RL nodes — candidate for tightening (see coherence)."))


_seed_catalog_v0_1()


# ---------------------------------------------------------------- migrations
# from_version -> (to_version, upgrade_fn). Each is one forward step; chains are
# walked by migrate_belief. The fn must return a Belief (version is then enforced).
_MIGRATIONS: Dict[str, Tuple[str, Callable[[Belief], Belief]]] = {}


def register_migration(from_version: str, to_version: str,
                       fn: Callable[[Belief], Belief]) -> None:
    if from_version == to_version:
        raise ValueError("a migration must change the version")
    _MIGRATIONS[from_version] = (to_version, fn)


def has_migration_path(from_version: str, to_version: str) -> bool:
    cur, seen = from_version, set()
    while cur != to_version:
        if cur in seen or cur not in _MIGRATIONS:
            return False
        seen.add(cur)
        cur = _MIGRATIONS[cur][0]
    return True


def migrate_belief(belief: Belief, target: str = INTERLINGUA_VERSION) -> Belief:
    """Upgrade ``belief`` forward through registered migrations until it speaks
    ``target``. No-op if it already does. Raises if no path exists."""
    cur, seen = belief, set()
    while cur.schema_version != target:
        v = cur.schema_version
        if v in seen or v not in _MIGRATIONS:
            raise InterlinguaError(
                f"no migration path from v{v} to v{target} for {cur.type!r} belief")
        seen.add(v)
        to_v, fn = _MIGRATIONS[v]
        cur = fn(cur)
        if cur.schema_version != to_v:        # enforce the version bump
            cur = replace(cur, schema_version=to_v)
    return cur


# Example migration 0.0 -> 0.1 (proves the mechanism; a real future bump registers
# its own). v0.0 forecasts used the key "value"; v0.1 renamed it to "estimate".
def _migrate_0_0_to_0_1(b: Belief) -> Belief:
    payload = dict(b.payload)
    if b.type == "forecast" and "value" in payload:
        payload["estimate"] = payload.pop("value")
    return replace(b, payload=payload, schema_version="0.1")


register_migration("0.0", "0.1", _migrate_0_0_to_0_1)


# ------------------------------------------------------------ conformance/drift
@dataclass
class Violation:
    """One way a belief fell off the interlingua contract."""
    source: str
    belief_type: str
    kind: str            # unknown_type | missing_keys | version_mismatch
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "belief_type": self.belief_type,
                "kind": self.kind, "detail": self.detail}


def validate_belief(belief: Belief, *, allow_migratable: bool = True) -> List[Violation]:
    """Check one belief against the current interlingua. Returns the violations
    (empty == conforms). A version mismatch with a registered migration path is
    NOT a violation (the belief is migratable); without one, it is."""
    viols: List[Violation] = []
    schema = _CATALOG.get(belief.type)
    if schema is None:
        viols.append(Violation(belief.source, belief.type, "unknown_type",
                               f"{belief.type!r} not in interlingua v{INTERLINGUA_VERSION}"))
        return viols  # can't key-check an unknown type
    checked = belief
    if belief.schema_version != INTERLINGUA_VERSION:
        if allow_migratable and has_migration_path(belief.schema_version, INTERLINGUA_VERSION):
            checked = migrate_belief(belief, INTERLINGUA_VERSION)  # check the upgraded payload
        else:
            viols.append(Violation(
                belief.source, belief.type, "version_mismatch",
                f"belief v{belief.schema_version} != interlingua v{INTERLINGUA_VERSION}"
                " and no migration path"))
    missing = [k for k in schema.required_keys if k not in checked.payload]
    if missing:
        viols.append(Violation(belief.source, belief.type, "missing_keys",
                               f"missing required key(s) {missing}"))
    return viols


def detect_drift(beliefs: Sequence[Belief], *, allow_migratable: bool = True) -> List[Violation]:
    """All conformance violations across a belief stream (e.g. a PathwayResult)."""
    out: List[Violation] = []
    for b in beliefs:
        out.extend(validate_belief(b, allow_migratable=allow_migratable))
    return out


@dataclass
class ConformanceReport:
    version: str
    n_beliefs: int
    n_conforming: int
    violations: List[Violation] = field(default_factory=list)

    @property
    def conforms(self) -> bool:
        return not self.violations

    def by_kind(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for v in self.violations:
            out[v.kind] = out.get(v.kind, 0) + 1
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "n_beliefs": self.n_beliefs,
            "n_conforming": self.n_conforming,
            "conforms": self.conforms,
            "by_kind": self.by_kind(),
            "violations": [v.to_dict() for v in self.violations],
        }


def conformance_report(beliefs: Sequence[Belief], *, allow_migratable: bool = True) -> ConformanceReport:
    beliefs = list(beliefs)
    viols = detect_drift(beliefs, allow_migratable=allow_migratable)
    bad_sources = {(v.source, v.belief_type) for v in viols}
    n_conforming = sum(1 for b in beliefs if (b.source, b.type) not in bad_sources)
    return ConformanceReport(INTERLINGUA_VERSION, len(beliefs), n_conforming, viols)


# --------------------------------------------------------------- coherence
@dataclass
class CoherenceNote:
    """A soft finding: one belief type whose payload differs across its emitters."""
    belief_type: str
    sources: List[str]
    common_keys: List[str]
    divergent_keys: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {"belief_type": self.belief_type, "sources": list(self.sources),
                "common_keys": list(self.common_keys),
                "divergent_keys": list(self.divergent_keys)}


def interlingua_coherence(beliefs: Sequence[Belief]) -> List[CoherenceNote]:
    """Report belief types whose payload keys are inconsistent across the nodes
    that emit them — the early sign that the interlingua needs tightening at the
    next version (distinct from a hard catalog violation)."""
    by_type: Dict[str, List[Belief]] = {}
    for b in beliefs:
        by_type.setdefault(b.type, []).append(b)
    notes: List[CoherenceNote] = []
    for t, group in sorted(by_type.items()):
        sources = sorted({b.source for b in group})
        if len(sources) < 2:
            continue
        keysets = [set(b.payload) for b in group]
        common = set.intersection(*keysets) if keysets else set()
        union = set.union(*keysets) if keysets else set()
        divergent = union - common
        if divergent:
            notes.append(CoherenceNote(t, sources, sorted(common), sorted(divergent)))
    return notes
