# Changelog

All notable public disclosures for Matryoshka Continuity Weights are recorded here.

## [1.3.1] — 2026-08-28

### Clarified

- `PRIOR_ART.md` — added "Separate parameter bodies outside the compute graph
  (Qwen 3.8-Flash-Next / Qwen 4 preview, Alibaba, 2026)": the N-gram parameter
  body is attributed to Alibaba and discussed as adjacent work. No identity
  with the Matryoshka principle is claimed in either direction.
- `BENCHMARKS.md` — added a Conditions and limitations block: dataset, metric
  definition (recall@0.8), seed, hardware, absence of end-to-end baselines,
  and the scope of the reported figures.

## [1.3.0] — 2026-08-27

### Added

- `BENCHMARKS.md` — first public experimental evidence: a 1.074 GB plastic
  parameter substrate; fact writes at 12.5 ms p50 / 23.9 ms p99 on CPU only;
  Q/A recall 100% over a 1000-turn dialogue simulation; temporal versioning
  3/3; persistence verified; capacity ~8000 facts per GB without
  consolidation. Includes the economics comparison: rank-1 fact writes are
  ~10^5–10^6 cheaper in FLOPs than gradient-based online updates and are
  independent of the size of the stable core.
- MMI daemon technical specification (internal, `MMI/MMI-DEMON-TZ.md`):
  production form of the memory interface between dialogue turns.

### Note

- Experimental code and addressing/consolidation mechanisms remain local
  (see RESEARCH_SCOPE.md).

## [1.2.1] — 2026-08-27

### Clarified

- `PRIOR_ART.md` — added the classical mechanistic lineage: linear
  associative memory (Anderson 1972, Kohonen 1972), fast-weight
  programmers (Schmidhuber 1992), and the modern line (linear
  transformers, DeltaNet, test-time training, Titans/MIRAS). No new
  technical mechanisms disclosed.

## [1.2.0] — 2026-08-27

### Added

- `ANATOMY.md` — public conceptual architectural description of Matryoshka:
  the anatomy of autobiographical continuity (K + Φ(t)), nesting of plastic
  capacity across timescales, and an ordinary technical memory interface.
  No implementation-specific mechanisms are disclosed.

## [1.1.0] — 2026-08-27

### Added

- Boundary document (`BOUNDARY.md`) — Matryoshka is a memory substrate,
  not a governance layer; deployment-environment separation.
- Glossary of canonical terms (`GLOSSARY.md`).

### Clarified

- `MANIFEST.md` — explicit non-goals; stable-core revision marked as an
  owner-initiated procedure outside Matryoshka.
- `RESEARCH_SCOPE.md` — deployment-environment boundary.
- Root `README.md` — AURA-Retrieval architecture and Matryoshka Continuity
  Weights separated; research-10…14 marked as early exploratory materials
  with preliminary, non-canonical terminology.

### Fixed

- `CITATION.cff` license metadata aligned to Apache-2.0 for published content;
  undisclosed implementation details remain unpublished and unlicensed.



### Added

- Canonical public definition of Matryoshka Continuity Weights.
- Core distinction between a stable cognitive core \(K\) and self-authored plastic parameter memory \(\Phi(t)\).
- Definition of the passive plastic-substrate principle.
- Definition of nested personal-memory timescales.
- Definition of temporal and version-aware autobiographical continuity.
- Public scope and non-disclosure boundary.
- Related-work and distinctions statement.
- Citation metadata.

### Status

Initial public conceptual disclosure.
