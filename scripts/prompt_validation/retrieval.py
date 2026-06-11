"""Two-stage recall routing + hybrid scoring prototype.

Validates the new recall design on synthetic fixtures:

  Stage 1 (collection routing): each collection's ``inclusion`` flag
  decides whether it participates at all.  ``always`` is in unconditionally
  (skills); ``relevant`` is in iff the message embed-matches the
  collection's content-reflective description above a threshold; ``never``
  is out.  This is where the prompt-shortening comes from and replaces the
  per-entry noise floor.

  Stage 2 (entity retrieval): within included collections, rank entries by
  a hybrid of embedding cosine and IDF-weighted lexical coverage (fused
  with reciprocal-rank fusion) and take the top-N.  No floor — the
  collection-level gate already decided relevance.

Run from repo root::

    PYTHONPATH=. uv run --python 3.12 --with openai \
        python scripts/prompt_validation/retrieval.py
"""
from __future__ import annotations

from similarity.embeddings import cosine_similarity
from similarity.lexical import idf, lexical_coverage, reciprocal_rank_fusion, tokens

from scripts.prompt_validation._harness import CaseResult, Harness, load_seed_skills
from scripts.prompt_validation.fixtures import MESSAGES, SYNTH_COLLECTIONS

NAME = "retrieval"

RECALL_LIMIT = 5  # mirrors production RECALL_LIMIT
STAGE1_THRESHOLD = 0.40  # description-anchor inclusion gate


def max_cos(anchor_vecs: list[list[float]], vec: list[float]) -> float:
    """Best cosine across the conversation-window anchors.

    Mirrors production's ``max(weighted_decay_over_history, cosine_to_current)``
    — a topic-less follow-up ("keep researching") still matches via an
    earlier turn that named the topic.
    """
    return max(cosine_similarity(a, vec) for a in anchor_vecs)


def hybrid_rank(
    query_text: str,
    anchor_vecs: list[list[float]],
    docs: list[tuple[str, str, list[float]]],  # (key, text, vec)
) -> list[str]:
    """Rank doc keys by RRF of (best-anchor cosine) and IDF-lexical.

    Uses the same ``similarity.lexical`` primitives production composes in
    ``MemoryStore._rank_hybrid`` — the harness validates the real scoring
    code, not a parallel reimplementation.
    """
    query_tokens = tokens(query_text)
    idf_map = idf([tokens(text) for _, text, _ in docs])
    cos_rank = sorted(docs, key=lambda d: -max_cos(anchor_vecs, d[2]))
    lex_rank = sorted(
        docs, key=lambda d: -lexical_coverage(query_tokens, tokens(d[1]), idf_map)
    )
    return reciprocal_rank_fusion([[d[0] for d in cos_rank], [d[0] for d in lex_rank]])


def run(h: Harness, samples: int = 1, only: str | None = None) -> list[CaseResult]:
    """Registry entry point. Embeddings are deterministic, so ``samples`` is
    ignored (single pass). Returns one CaseResult per message covering both
    the stage-1 routing and the stage-2 skill-surfacing checks."""
    seed_skills, _ = load_seed_skills()
    skill_vecs = h.embed([f"{k}\n{c}" for k, c in seed_skills])
    skill_docs = [(k, f"{k}\n{c}", v) for (k, c), v in zip(seed_skills, skill_vecs)]
    desc_vecs = {c.name: v for c, v in zip(
        SYNTH_COLLECTIONS, h.embed([c.description for c in SYNTH_COLLECTIONS]))}

    results: list[CaseResult] = []
    for msg in MESSAGES:
        if only and only != msg.id:
            continue
        fails: list[str] = []
        anchor_text = " ".join([*msg.history, msg.text])
        anchor_vecs = h.embed([msg.text, *msg.history])

        included = set()
        for col in SYNTH_COLLECTIONS:
            if col.inclusion == "always":
                included.add(col.name)
            elif col.inclusion == "relevant" and max_cos(anchor_vecs, desc_vecs[col.name]) >= STAGE1_THRESHOLD:
                included.add(col.name)
        for exp in msg.collections:
            if exp not in included:
                fails.append(f"routing: expected '{exp}' included")
        for col in SYNTH_COLLECTIONS:
            if col.inclusion == "relevant" and col.name not in msg.collections and col.name in included:
                fails.append(f"routing: '{col.name}' should have dropped")

        if msg.skill is not None:
            ranked = hybrid_rank(anchor_text, anchor_vecs, skill_docs)[:RECALL_LIMIT]
            if msg.skill not in ranked:
                fails.append(f"skill '{msg.skill}' not in top-{RECALL_LIMIT}")

        results.append(CaseResult(f"{NAME}:{msg.id}", not fails, fails))
    return results


def main() -> None:
    h = Harness()
    seed_skills, _ = load_seed_skills()

    # Embed skills (key+content as the doc), collection descriptions, entries.
    skill_vecs = h.embed([f"{k}\n{c}" for k, c in seed_skills])
    skill_docs = [(k, f"{k}\n{c}", v) for (k, c), v in zip(seed_skills, skill_vecs)]

    desc_vecs = {c.name: v for c, v in zip(SYNTH_COLLECTIONS, h.embed([c.description for c in SYNTH_COLLECTIONS]))}
    entry_docs: dict[str, list[tuple[str, str, list[float]]]] = {}
    for col in SYNTH_COLLECTIONS:
        vs = h.embed(list(col.entries))
        entry_docs[col.name] = [(e, e, v) for e, v in zip(col.entries, vs)]

    n_skill_ok = n_skill_tot = 0
    n_route_ok = n_route_tot = 0

    for msg in MESSAGES:
        # Anchor on the conversation window (current message + recent history).
        anchor_text = " ".join([*msg.history, msg.text])
        anchor_vecs = h.embed([msg.text, *msg.history])
        print(f"\n{'=' * 78}\n## {msg.id}\n{'=' * 78}\n  msg: {msg.text[:88]}")

        # ── Stage 1: collection routing ────────────────────────────────────
        included, dropped = [], []
        for col in SYNTH_COLLECTIONS:
            if col.inclusion == "always":
                included.append(col.name)
            elif col.inclusion == "never":
                dropped.append(col.name)
            else:  # relevant
                sim = max_cos(anchor_vecs, desc_vecs[col.name])
                (included if sim >= STAGE1_THRESHOLD else dropped).append(f"{col.name}({sim:.2f})")
        print(f"  stage-1 include: {included}")
        print(f"  stage-1 drop:    {dropped}")

        # Score routing: expected topical collections must be included.
        for exp in msg.collections:
            n_route_tot += 1
            if any(exp == x.split("(")[0] for x in included):
                n_route_ok += 1
            else:
                print(f"  !! ROUTING MISS: expected '{exp}' included")
        # Non-expected topical collections must be dropped.
        for col in SYNTH_COLLECTIONS:
            if col.name not in msg.collections and col.inclusion == "relevant":
                n_route_tot += 1
                if any(col.name == x.split("(")[0] for x in dropped):
                    n_route_ok += 1
                else:
                    print(f"  !! ROUTING FALSE-INCLUDE: '{col.name}' should have dropped")

        # ── Stage 2: skills retrieval (always-on) ─────────────────────────
        ranked = hybrid_rank(anchor_text, anchor_vecs, skill_docs)[:RECALL_LIMIT]
        print(f"  skills top-{RECALL_LIMIT}: {ranked}")
        if msg.skill is not None:
            n_skill_tot += 1
            if msg.skill in ranked:
                n_skill_ok += 1
                print(f"  skill OK: '{msg.skill}' surfaced (rank {ranked.index(msg.skill) + 1})")
            else:
                print(f"  !! SKILL MISS: '{msg.skill}' not in top-{RECALL_LIMIT}")

        # ── Stage 2: entries of included topical collections ──────────────
        for exp in msg.collections:
            if exp in entry_docs:
                top = hybrid_rank(anchor_text, anchor_vecs, entry_docs[exp])[:3]
                print(f"  {exp} top entries: {[t[:40] for t in top]}")

    print(f"\n{'=' * 78}\n# Summary (threshold={STAGE1_THRESHOLD})\n{'=' * 78}")
    print(f"  skill surfaced in top-{RECALL_LIMIT}: {n_skill_ok}/{n_skill_tot}")
    print(f"  stage-1 routing correct:        {n_route_ok}/{n_route_tot}")
    print(f"  {h.metrics.summary()}")


if __name__ == "__main__":
    main()
