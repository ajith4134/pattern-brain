"""Phase 8 — slice 8: the LLM experiment-PLANNER (PLAN.md §12).

Closes the loop the owner's pasted chat emphasized: the project's OWN LLM reads the
leaderboard and proposes the next network combinations to try — as a CONTROLLER,
never the scorer. The golden rule (AIDE/MLE-STAR/AutoML-Agent, 2025): the LLM
proposes ``DAGSpec`` genomes; accuracy comes ONLY from ``StackedDAG.score`` +
the ``Evaluator`` DSR gate inside ``DAGSearch``. The LLM cannot set a score, and
any proposal referencing a node outside the pool is dropped.

Works fully offline: with no LLM it falls back to a heuristic proposer (mutations /
crossovers of the leaderboard's best, reusing the slice-7 genome operators). The
LLM boundary is one injected ``llm_chat(messages) -> str`` callable (the
``llm.auto_text_completer`` chain or a fake for tests).

Domain-agnostic (Rule 23): proposes over generic node_types.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Optional

from ..network import DAGSpec
from ..search import DAGSearch, COMBINERS

LLMChat = Callable[[List[Dict[str, str]]], str]


class DAGPlanner:
    """Proposes Stacked-DAG experiments (LLM-led or heuristic); the search scores them."""

    def __init__(self, search: DAGSearch, llm_chat: Optional[LLMChat] = None) -> None:
        self.search = search
        self.llm_chat = llm_chat

    # ------------------------------------------------------------------ prompt
    def _state(self, k: int = 8) -> Dict[str, object]:
        top = self.search.lb.top(k, metric="crps_skill", dataset_id=self.search.dataset_id)
        return {
            "pool": self.search.pool,
            "combiners": COMBINERS,
            "max_base": self.search.max_base,
            "summary": self.search.lb.summary(dataset_id=self.search.dataset_id),
            "leaderboard": [{"layers": json.loads(r["spec_json"])["layers"],
                             "combiner": r["combiner"], "skill": r["crps_skill"],
                             "dsr": r["dsr"]} for r in top],
        }

    def _messages(self, n: int) -> List[Dict[str, str]]:
        st = self._state()
        system = (
            "You are an ML architecture planner. You PROPOSE stacked-ensemble DAGs to try; "
            "you NEVER report accuracy — a separate validator scores them. A DAG is "
            '{"layers": [[node, ...]], "combiner": "mixture"|"stacked"}. Use ONLY nodes from '
            "the given pool. Propose diverse, plausible combinations that might beat the "
            "current leaderboard. Reply with ONLY a JSON array of DAG objects.")
        user = (f"pool={st['pool']}\ncombiners={st['combiners']}\nmax_base={st['max_base']}\n"
                f"current_leaderboard(best first)={json.dumps(st['leaderboard'])}\n"
                f"Propose {n} new DAGs as a JSON array.")
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # ----------------------------------------------------------------- propose
    def propose(self, n: int = 4) -> List[DAGSpec]:
        if self.llm_chat is not None:
            try:
                specs = [s for s in self._parse(self.llm_chat(self._messages(n))) if self._valid(s)]
                if specs:
                    return specs[:n]
            except Exception:
                pass
        return self._heuristic(n)

    def _heuristic(self, n: int) -> List[DAGSpec]:
        best = self.search.lb.best(dataset_id=self.search.dataset_id)
        seeds: List[DAGSpec] = []
        if best:
            seeds.append(DAGSpec.from_dict(json.loads(best["spec_json"])))
        out: List[DAGSpec] = []
        while len(out) < n:
            if len(seeds) >= 2 and self.search.rng.random() < 0.4:
                out.append(self.search._crossover(self.search.rng.choice(seeds),
                                                  self.search.rng.choice(seeds)))
            elif seeds and self.search.rng.random() < 0.7:
                out.append(self.search._mutate(self.search.rng.choice(seeds)))
            else:
                out.append(self.search._random_spec())
        return out

    # ------------------------------------------------------------------- parse
    def _parse(self, text: str) -> List[DAGSpec]:
        m = re.search(r"\[.*\]", text, re.DOTALL)        # first JSON array in the reply
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except Exception:
            return []
        specs = []
        for d in data if isinstance(data, list) else []:
            if not isinstance(d, dict):
                continue
            layers = d.get("layers") or ([d["base"]] if "base" in d else None)
            if not layers:
                continue
            # keep only pool nodes; drop empty layers
            clean = [[nd for nd in layer if nd in self.search.pool] for layer in layers]
            clean = [layer for layer in clean if layer]
            if not clean:
                continue
            combiner = d.get("combiner", "mixture")
            combiner = combiner if combiner in COMBINERS else "mixture"
            specs.append(DAGSpec(clean, combiner))
        return specs

    def _valid(self, spec: DAGSpec) -> bool:
        return (bool(spec.layers) and all(layer for layer in spec.layers)
                and all(nd in self.search.pool for layer in spec.layers for nd in layer)
                and spec.combiner in COMBINERS)

    # ------------------------------------------------------------- plan + score
    def plan_and_search(self, rounds: int = 3, n_per_round: int = 4) -> Optional[Dict]:
        """LLM (or heuristic) proposes → DAGSearch SCORES + records → repeat. The
        leaderboard grows each round so the next proposals see the latest state."""
        self.search._budget = max(self.search._budget, rounds * n_per_round)
        for _ in range(rounds):
            for spec in self.propose(n_per_round):
                self.search._score_and_record(spec)      # only the search/Evaluator scores
        return self.search.best()


__all__ = ["DAGPlanner"]
