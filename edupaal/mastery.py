"""The EduPAAL mastery engine: versioned, explainable, deterministic heuristics.

PAAL *owns* mastery transitions — it is not dumb storage. Vertical agents
report normalized evidence; the engine applies the promotion rules. Every
rule parameter is a tunable knob (see ``MasteryParams``): the framework
defines the shape of the rules and ships transparent defaults, the
deployment controls the dials.

heuristic-v1 rules
------------------
* Evidence and assertions target *leaf* TOPIC nodes only (enforced — a
  decomposed topic must be reported through its sub-topics, the finest grain
  available). Concept/subject/space mastery, and the mastery of a decomposed
  topic, is pure rollup (see ``edupaal.retrieval``). Overrides may target any
  node and win over rollup for that node.
* The first ever evidence for a topic moves UNKNOWN -> BEGINNER: any
  engagement establishes the beginner state.
* From BEGINNER (or INTERMEDIATE), the engine evaluates the up-to-K most
  recent evidence items inside a W-day window anchored at the latest
  evidence timestamp:
    - need at least K items in the set,
    - mean performance must clear the bar for the next level
      (t_intermediate / t_advanced),
    - no item in the set may fall below t_contradict (it blocks promotion),
    - INTERMEDIATE -> ADVANCED additionally requires >= 2 distinct
      activity_types in the set when cross_modal_advanced is on.
* Promotions chain within one call (BEGINNER -> INTERMEDIATE -> ADVANCED),
  each step writing its own MasteryRecord. Demotion/decay is intentionally
  out of scope for v1.

Every transition writes a MasteryRecord carrying the rule version, the exact
parameters in effect, and the evidence IDs that caused it — so any past
decision can be replayed and explained.

Vertical tracks and quorum aggregation
--------------------------------------
Evidence carries ``source_agent`` — the vertical that reported it — and every
MasteryRecord is scoped to one ``vertical_id`` (the same value). Each
vertical promotes its own track with its own thresholds (see
``MasteryEngine``'s ``vertical_params``); the shared (learner, topic) level
that rollup and retrieval see is derived by the engine's quorum function —
``aggregate_vertical_mastery`` by default, replaceable with any ``QuorumFn``
(see below):

* verticals at UNKNOWN are dropped (no evidence is not evidence of
  ignorance); all UNKNOWN -> overall UNKNOWN;
* overall ADVANCED needs ``xvertical_quorum_advanced`` (default 2)
  verticals attesting ADVANCED;
* a lone ADVANCED caps at INTERMEDIATE overall — the transfer bar:
  advancement must be confirmed across contexts;
* anything less falls back to the highest attested level (so one
  INTERMEDIATE, or one BEGINNER, reports as-is).
* Privileged assertions bypass the quorum: an assertion is a deliberate
  statement about the learner, not a noisy heuristic inference, so the
  latest assertion floors the shared level. Assertions are learner-level
  knowledge — they are excluded from per-vertical tracks and never count
  as a vertical's attestation.

The quorum itself is governed by the *shared* parameters: the learning
plan's override for the node (or nearest ancestor's), else the engine
defaults. Per-vertical params tune each vertical's own promotion bars,
not the aggregation.

Deployments may replace the whole transfer rule: pass a ``QuorumFn`` to
``MasteryEngine`` (or ``EduPAALSkill``). It receives each vertical's
heuristic track levels plus the resolved shared params, and must be pure
and deterministic; the assertion floor is applied afterwards by the
engine, so custom logic composes with privileged assertions.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from .entities import (
    DynamicOverride,
    Evidence,
    KnowledgeNode,
    MASTERY_SCORES,
    MasteryLevel,
    MasteryParams,
    MasteryRecord,
    NodeLevel,
    _new_id,
    _utcnow,
)
from .graph import KnowledgeGraph
from .store import StorageBackend

HEURISTIC_VERSION = "heuristic-v1"
ASSERTION_VERSION = "assertion-v1"


def aggregate_vertical_mastery(
    levels: Mapping[str, MasteryLevel], params: MasteryParams
) -> MasteryLevel:
    """Derive the shared (learner, topic) level from per-vertical levels.

    Pure and deterministic: the same per-vertical levels always aggregate
    the same way. ``levels`` maps vertical_id -> that vertical's current
    *heuristic* level (UNKNOWN allowed — it means "this vertical has no
    signal"; assertion records are excluded, see ``vertical_levels``).

    The transfer bar: overall ADVANCED needs ``xvertical_quorum_advanced``
    (default 2) verticals attesting ADVANCED. A lone ADVANCED caps at
    INTERMEDIATE — advancement must be confirmed across contexts. Anything
    less falls back to the highest attested level.
    """
    attested = {v: l for v, l in levels.items() if l != MasteryLevel.UNKNOWN}
    if not attested:
        return MasteryLevel.UNKNOWN
    n_advanced = sum(1 for l in attested.values() if l == MasteryLevel.ADVANCED)
    if n_advanced >= params.xvertical_quorum_advanced:
        return MasteryLevel.ADVANCED
    if n_advanced >= 1:
        # Cross-context confirmation failed: one vertical's ADVANCED is
        # INTERMEDIATE overall, no matter how many verticals attest below.
        return MasteryLevel.INTERMEDIATE
    return max(attested.values(), key=lambda l: MASTERY_SCORES[l])


# The quorum contract. A quorum function derives the shared (learner, topic)
# level from every vertical's current *heuristic* track level, plus the
# resolved shared MasteryParams (plan override -> engine defaults).
#
# Contract for a custom function:
#   * pure and deterministic: same (levels, params) -> same level, no I/O,
#     no randomness, no wall-clock reads;
#   * ``levels`` maps vertical_id -> that vertical's heuristic level.
#     UNKNOWN means "no signal" and assertions are already excluded — a
#     custom function never sees assertion records and cannot suppress
#     them: the engine applies the assertion floor *after* the quorum
#     function runs (see MasteryEngine._current_level);
#   * honor ``params`` where it makes sense (e.g. read
#     params.xvertical_quorum_advanced instead of hardcoding a bar) so the
#     deployment's knobs keep working.
QuorumFn = Callable[[Mapping[str, MasteryLevel], MasteryParams], MasteryLevel]


def _require_node(graph: KnowledgeGraph, node_id: str) -> KnowledgeNode:
    """Fetch a node or fail loudly: an unknown node id is a caller bug and
    must never masquerade as an unevaluated (UNKNOWN) node."""
    try:
        return graph.get(node_id)
    except KeyError:
        raise ValueError(f"unknown node: {node_id}") from None


class MasteryEngine:
    def __init__(
        self,
        store: StorageBackend,
        graph: KnowledgeGraph,
        default_params: Optional[MasteryParams] = None,
        vertical_params: Optional[Dict[str, MasteryParams]] = None,
        quorum_fn: Optional[QuorumFn] = None,
    ) -> None:
        self.store = store
        self.graph = graph
        self.default_params = default_params or MasteryParams()
        # Per-vertical threshold overrides, keyed by vertical_id (== the
        # Evidence.source_agent that vertical reports under). Lets a quest
        # vertical promote on looser bars than a tutor vertical, for example.
        self.vertical_params = dict(vertical_params or {})
        # How per-vertical tracks combine into the shared (learner, topic)
        # level. Defaults to aggregate_vertical_mastery; pass a QuorumFn to
        # define the deployment's own transfer rule.
        self.quorum_fn = quorum_fn or aggregate_vertical_mastery

    # ------------------------------------------------------------ parameters

    def params_for(
        self,
        learner_id: str,
        node_id: str,
        vertical_id: str = "default",
    ) -> MasteryParams:
        """Resolve the effective knobs for one vertical's track.

        Precedence, most specific first: the plan's override for the node,
        the nearest ancestor's override (concept, then subject...), the
        vertical's own params, else the engine defaults.
        """
        plan = self.store.get_plan_for_learner(learner_id)
        if plan is not None:
            node = _require_node(self.graph, node_id)
            candidates = [node_id] + [a.id for a in self.graph.ancestors(node_id)]
            for cid in candidates:
                if cid in plan.criteria_overrides:
                    return plan.criteria_overrides[cid]
        return self.vertical_params.get(vertical_id, self.default_params)

    # ---------------------------------------------------------------- record

    def record_evidence(self, evidence: Evidence) -> List[MasteryRecord]:
        """Persist evidence and apply the promotion rules.

        Returns the MasteryRecords written by this call (possibly empty when
        the evidence does not move mastery). Raises on invalid evidence.
        """
        node = _require_node(self.graph, evidence.node_id)
        if node.level != NodeLevel.TOPIC:
            raise ValueError(
                f"evidence must target a TOPIC node ({node.id} is {node.level.value}); "
                "higher levels roll up from topics"
            )
        if self.graph.children(node.id):
            raise ValueError(
                f"topic {node.id} is decomposed into sub-topics; record evidence "
                "on the finest-grained sub-topic instead"
            )
        self.store.save_evidence(evidence)
        # Promotion is per-vertical: the vertical that reported this evidence
        # (evidence.source_agent) promotes its own track from its own
        # evidence slice, with its own thresholds. The shared (learner,
        # topic) level is derived afterwards by quorum aggregation.
        vertical_id = evidence.source_agent

        written: List[MasteryRecord] = []
        if (
            self._current_level(evidence.learner_id, evidence.node_id, vertical_id)
            == MasteryLevel.UNKNOWN
        ):
            written.append(
                self._write_record(
                    evidence.learner_id,
                    evidence.node_id,
                    MasteryLevel.BEGINNER,
                    HEURISTIC_VERSION,
                    self.params_for(
                        evidence.learner_id, evidence.node_id, vertical_id
                    ),
                    [evidence.id],
                    updated_at=evidence.occurred_at,
                    vertical_id=vertical_id,
                )
            )

        # Fixed-point promotion: keep stepping while the evidence supports it.
        # Each step re-reads the stored level, so one strong evidence set can
        # chain BEGINNER -> INTERMEDIATE -> ADVANCED, writing one record per step.
        while True:
            level = self._current_level(
                evidence.learner_id, evidence.node_id, vertical_id
            )
            if level == MasteryLevel.ADVANCED:
                break
            nxt = self._try_promote(
                evidence.learner_id, evidence.node_id, level, vertical_id
            )
            if nxt is None:
                break
            written.append(nxt)
            # Fail-loud termination guard: a promotion write must advance the
            # observable level. Records are ordered by (updated_at, rowid) and
            # _try_promote stamps max(evidence time, now), so a freshly written
            # record always sorts last. If the level did not move, re-looping
            # would spin forever writing duplicates — surface the ordering
            # violation instead of hanging.
            if (
                self._current_level(evidence.learner_id, evidence.node_id, vertical_id)
                == level
            ):
                raise RuntimeError(
                    f"promotion to {nxt.level.value} did not advance "
                    f"{evidence.node_id} past {level.value}; refusing to loop"
                )

        return written

    def _try_promote(
        self, learner_id: str, node_id: str, current: MasteryLevel, vertical_id: str
    ) -> Optional[MasteryRecord]:
        params = self.params_for(learner_id, node_id, vertical_id)
        evidence = sorted(
            (
                e
                for e in self.store.list_evidence(learner_id, node_id)
                if e.source_agent == vertical_id
            ),
            # (occurred_at, id): identical timestamps break ties by id so the
            # same evidence set always promotes the same way, regardless of
            # the order verticals submitted it in.
            key=lambda e: (e.occurred_at, e.id),
        )
        if not evidence:
            return None
        anchor = max(e.occurred_at for e in evidence)
        cutoff = anchor - timedelta(days=params.window_days)
        windowed = [e for e in evidence if e.occurred_at >= cutoff]
        candidates = windowed[-params.k_evidence :]
        if len(candidates) < params.k_evidence:
            return None

        if current == MasteryLevel.BEGINNER:
            target, bar = MasteryLevel.INTERMEDIATE, params.t_intermediate
        elif current == MasteryLevel.INTERMEDIATE:
            target, bar = MasteryLevel.ADVANCED, params.t_advanced
        else:  # pragma: no cover - loop guard makes this unreachable
            return None

        mean_perf = sum(e.performance for e in candidates) / len(candidates)
        if mean_perf < bar:
            return None
        if any(e.performance < params.t_contradict for e in candidates):
            return None
        if (
            target == MasteryLevel.ADVANCED
            and params.cross_modal_advanced
            and len({e.activity_type for e in candidates}) < 2
        ):
            return None

        # Stamp the later of the evidence anchor and now. Evidence-driven
        # records keep evidence-logical time when the evidence is current
        # (replay stays deterministic), but a record computed *now* must never
        # sort before a wall-clock assertion written earlier: records are
        # ordered by (updated_at, rowid), and a promotion that does not become
        # current would make the fixed-point loop in record_evidence spin
        # forever writing duplicates.
        return self._write_record(
            learner_id,
            node_id,
            target,
            HEURISTIC_VERSION,
            params,
            [e.id for e in candidates],
            updated_at=max(anchor, _utcnow()),
            vertical_id=vertical_id,
        )

    # --------------------------------------------------------------- assert

    def assert_mastery(
        self,
        learner_id: str,
        node_id: str,
        level: MasteryLevel,
        asserted_by: str,
        reason: str,
        vertical_id: str = "default",
    ) -> MasteryRecord:
        """Privileged override path for assessment agents and humans.

        Writes a MasteryRecord marked as an assertion. It does not erase or
        rewrite history — it appends, like everything else.

        Assertions are learner-level knowledge, not track knowledge: the
        assertion record is excluded from per-vertical tracks (see
        ``vertical_levels``) and instead floors the shared (learner, topic)
        level via ``_current_level``, bypassing the quorum — a privileged
        assertion is a deliberate statement about the learner, not a noisy
        heuristic inference, so it needs no cross-vertical confirmation.
        Subsequent evidence continues to accumulate on the vertical tracks
        underneath the floor.
        """
        if isinstance(level, str):
            level = MasteryLevel(level)
        if level == MasteryLevel.UNKNOWN:
            raise ValueError("cannot assert UNKNOWN")
        node = _require_node(self.graph, node_id)
        if node.level != NodeLevel.TOPIC:
            raise ValueError("assertions target TOPIC nodes; higher levels roll up")
        if self.graph.children(node_id):
            raise ValueError(
                f"topic {node_id} is decomposed into sub-topics; assert on the "
                "finest-grained sub-topic instead (or use an override)"
            )
        return self._write_record(
            learner_id,
            node_id,
            level,
            ASSERTION_VERSION,
            self.params_for(learner_id, node_id, vertical_id),
            evidence_ids=[],
            assertion=True,
            asserted_by=asserted_by,
            reason=reason,
            updated_at=_utcnow(),
            vertical_id=vertical_id,
        )

    # -------------------------------------------------------------- resolve

    def current_level(self, learner_id: str, node_id: str) -> MasteryLevel:
        """The shared (learner, topic) level: the engine's quorum function
        over every vertical's track (default ``aggregate_vertical_mastery``;
        see ``QuorumFn`` for the override contract).

        For aggregated views (concepts, subjects, decomposed topics) use
        ``effective_mastery`` / ``rolled_up_mastery`` instead. For one
        vertical's own track, use ``vertical_level``.
        """
        return self._current_level(learner_id, node_id)

    def _current_level(
        self, learner_id: str, node_id: str, vertical_id: Optional[str] = None
    ) -> MasteryLevel:
        if vertical_id is not None:
            # One vertical's own track: a pure function of that vertical's
            # heuristic evidence. Assertions are learner-level knowledge, not
            # track knowledge, so they do not appear here — see
            # vertical_levels; the assertion floor below handles them.
            return self.vertical_level(learner_id, node_id, vertical_id)
        levels = self.vertical_levels(learner_id, node_id)
        agg = self.quorum_fn(
            levels,
            self.params_for(learner_id, node_id),
        )
        # Assertions bypass the quorum: a privileged assertion is a deliberate
        # statement about the learner, not a noisy heuristic inference, so it
        # needs no cross-vertical confirmation. The latest assertion (history
        # is oldest -> newest) floors the shared level.
        floor = self._assertion_floor(learner_id, node_id)
        if floor is None:
            return agg
        if agg == MasteryLevel.UNKNOWN:
            return floor
        return max((agg, floor), key=lambda l: MASTERY_SCORES[l])

    def _assertion_floor(
        self, learner_id: str, node_id: str
    ) -> Optional[MasteryLevel]:
        """Latest assertion level for (learner, node), or None.

        History is append-only and ordered oldest -> newest, so the last
        assertion seen wins — mirroring the single-track ratchet where the
        latest record wins.
        """
        floor: Optional[MasteryLevel] = None
        for rec in self.store.get_mastery_history(learner_id, node_id):
            if rec.assertion:
                floor = rec.level
        return floor

    def vertical_level(
        self, learner_id: str, node_id: str, vertical_id: str
    ) -> MasteryLevel:
        """One vertical's own track level: no aggregation, no rollup.

        Tracks are pure heuristic: assertion records are learner-level
        knowledge and never appear in a track (they floor the shared level
        instead — see ``_current_level``).
        """
        return self.vertical_levels(learner_id, node_id).get(
            vertical_id, MasteryLevel.UNKNOWN
        )

    def vertical_levels(
        self, learner_id: str, node_id: str
    ) -> Dict[str, MasteryLevel]:
        """Each vertical's current heuristic level for the node.

        History is ordered oldest -> newest, so the last record seen per
        vertical_id is its current level. Assertion records are excluded:
        an assertion is a privileged statement about the learner, not an
        inference from that vertical's evidence, and it must not count as
        the vertical's attestation in ``aggregate_vertical_mastery`` (it
        would otherwise double-count — once as a track attestation, once
        as the assertion floor).
        """
        levels: Dict[str, MasteryLevel] = {}
        for rec in self.store.get_mastery_history(learner_id, node_id):
            if rec.assertion:
                continue
            levels[rec.vertical_id] = rec.level
        return levels

    def active_override(
        self,
        learner_id: str,
        node_id: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> Optional[DynamicOverride]:
        """The override currently shaping a node.

        Precedence: an active node-scoped override beats an active global
        one. Among active overrides of the *same* scope, the most recently
        created wins — a newer operator correction supersedes an older one
        (a long-lived stale override must not silently shadow a newer short
        correction). Shadowed overrides remain in history as an audit trail.
        ``node_id=None`` matches only global overrides.
        """
        now = now or _utcnow()
        scoped: List[DynamicOverride] = []
        glob: List[DynamicOverride] = []
        for override in self.store.list_overrides(learner_id):
            if not override.is_active(now):
                continue
            if node_id is not None and override.scope_node_id == node_id:
                scoped.append(override)
            elif override.scope_node_id is None:
                glob.append(override)
        pool = scoped or glob
        if not pool:
            return None
        return max(pool, key=lambda o: o.created_at)

    def effective_mastery(
        self, learner_id: str, node_id: str, now: Optional[datetime] = None
    ) -> MasteryLevel:
        """Mastery as vertical agents should treat it: an active override
        (node-scoped, else global) wins; otherwise mastery rolls up the full
        chain below the node. Leaf nodes report their own latest record, so
        for leaf TOPIC nodes this is exactly the record history; for
        concepts/subjects/spaces (and decomposed topics) it is the rollup —
        never UNKNOWN-just-because-no-direct-record-exists."""
        return self.rolled_up_mastery(learner_id, node_id, now)[0]

    def rolled_up_mastery(
        self, learner_id: str, node_id: str, now: Optional[datetime] = None
    ) -> Tuple[MasteryLevel, Optional[float]]:
        """(level, mean_score) for any node at any depth, with full-chain
        rollup: sub-topic -> ... -> topic -> concept -> subject -> space.

        An explicit active override on the node always wins. Nodes with
        children report the mean of their children's scores (UNKNOWN children
        excluded; a node with no evaluated children is UNKNOWN). Leaf nodes
        report their own latest record. Score is None when UNKNOWN.
        Deterministic: score ties round to the lower level.
        """
        _require_node(self.graph, node_id)  # fail loudly on unknown ids
        override = self.active_override(learner_id, node_id, now)
        if override is not None:
            return override.level, float(MASTERY_SCORES[override.level])
        children = self.graph.children(node_id)
        if not children:
            level = self._current_level(learner_id, node_id)
            if level == MasteryLevel.UNKNOWN:
                return MasteryLevel.UNKNOWN, None
            return level, float(MASTERY_SCORES[level])
        scores: List[float] = []
        for child in children:
            _, s = self.rolled_up_mastery(learner_id, child.id, now)
            if s is not None:
                scores.append(s)
        if not scores:
            return MasteryLevel.UNKNOWN, None
        mean = sum(scores) / len(scores)
        nearest = min(
            MASTERY_SCORES.items(), key=lambda kv: (abs(kv[1] - mean), kv[1])
        )[0]
        return nearest, mean

    def promote_override(
        self, learner_id: str, scope_node_id: Optional[str] = None
    ) -> MasteryRecord:
        """Promote an active temporary override into a durable assertion.

        Promotes the *effective* override for the scope — the same one
        ``active_override`` reports (most recently created among active
        same-scope overrides). Promotion retires the scope's whole
        temporary stack: the effective override becomes the durable
        assertion and any shadowed same-scope overrides are retired with
        it, so the assertion stands on its own afterwards.
        """
        now = _utcnow()
        candidates = [
            o
            for o in self.store.list_overrides(learner_id)
            if o.is_active(now) and o.scope_node_id == scope_node_id
        ]
        if not candidates:
            raise ValueError("no active override for that scope to promote")
        target = max(candidates, key=lambda o: o.created_at)
        if target.scope_node_id is None:
            raise ValueError("global overrides cannot be promoted; scope them first")
        scope_node = _require_node(self.graph, target.scope_node_id)
        if scope_node.level != NodeLevel.TOPIC or self.graph.children(scope_node.id):
            # assert_mastery would reject this deep inside with a message about
            # "use an override" — which is exactly what we are promoting. Fail
            # here instead, with the actual reason.
            raise ValueError(
                f"cannot promote override on {scope_node.id} "
                f"({scope_node.level.value}): promotion writes a leaf-topic "
                "assertion, and only leaf TOPIC nodes can carry one; keep the "
                "temporary override instead"
            )
        record = self.assert_mastery(
            learner_id,
            target.scope_node_id,
            target.level,
            asserted_by="override-promotion",
            reason=f"promoted override {target.id}: {target.reason}",
        )
        # Retire the scope's whole temporary stack so the assertion stands
        # on its own: no active temporary override may contradict it.
        for o in candidates:
            o.promoted = True
            self.store.save_override(o)
        return record

    # --------------------------------------------------------------- internals

    def _write_record(
        self,
        learner_id: str,
        node_id: str,
        level: MasteryLevel,
        rule_version: str,
        params: MasteryParams,
        evidence_ids: List[str],
        updated_at: datetime,
        assertion: bool = False,
        asserted_by: Optional[str] = None,
        reason: Optional[str] = None,
        vertical_id: str = "default",
    ) -> MasteryRecord:
        record = MasteryRecord(
            id=_new_id("mr"),
            node_id=node_id,
            learner_id=learner_id,
            level=level,
            updated_at=updated_at,
            rule_version=rule_version,
            params_in_effect=params.as_dict(),
            evidence_ids=list(evidence_ids),
            assertion=assertion,
            asserted_by=asserted_by,
            reason=reason,
            vertical_id=vertical_id,
        )
        self.store.save_mastery_record(record)
        return record
