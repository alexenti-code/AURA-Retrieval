# Benchmarks and Economics of Parametric Fact Writing (v1.3.0)

**Status:** experimental evidence (public release). Measurements made on a
consumer laptop (Apple Silicon, CPU only, no GPU). Addressing and
consolidation mechanisms are intentionally not disclosed
(see [RESEARCH_SCOPE.md](RESEARCH_SCOPE.md)).

## What was measured

A plastic parameter substrate Phi of **1.074 GB** (32 associator regions,
d = 4096, fp16) was created as a single file. Facts are recorded as rank-1
state updates — no gradients, no loss function, no backpropagation through
any model. The stable cognitive core K is not touched and its size is
irrelevant to the cost of memory writes.

## Headline numbers

| Operation | Cost |
|---|---|
| Write one fact (hot path, in-RAM) | **12.5 ms p50 / 23.9 ms p99** |
| Write one fact (cold, incl. region load) | < 30 ms |
| Snapshot 1.074 GB substrate to disk | **363–377 ms** |
| Fact recall (single read) | ~12 ms |
| Dialogue run: 1000 turns, 2041 facts, 300 Q/A probes | recall@0.8 = **100%** |
| Temporal versioning (3 versions of one entity) | 3/3 readable, history retained |
| Persistence (full reload from disk) | recall preserved |
| Capacity | ~250 facts per region; ~8000 facts per 1 GB (no consolidation yet) |

## Economics: writing facts vs training weights

Cost model: FLOPs of one memory operation.

| Method to make a model "remember" | Compute per update | Hardware | Decoupled from core size? |
|---|---|---|---|
| Full fine-tuning on dialogue data | ~10^13–10^14 FLOP | GPU, seconds–hours | no — scales with model size |
| Online LoRA gradient step (7B, r=16, 1k tokens) | ~4·10^13 FLOP | GPU, seconds | partly |
| RAG: insert vector | ~10^2–10^3 FLOP | CPU, <1 ms | yes — but memory lives **outside** the model |
| **Matryoshka rank-1 write (this work)** | **~3.4·10^7 FLOP (measured 12.5 ms wall, CPU)** | **CPU only** | **yes — same cost for a 7B or a 70B core** |

The rank-1 fact write is **~10^5–10^6 times cheaper in FLOPs** than a
gradient-based online update, and its cost does not grow with the size of
the core. A write fits **between dialogue turns** (p99 < 31 ms), so an
agent can record what just happened in the same conversational pause where
a human would blink. Writes can be initiated by the model itself, per its
own decision that an event is worth remembering.

## What this does and does not show

Does show: parametric autobiographical memory of an agent can be written
durably, versionedly and cheaply on commodity hardware, on the fly, with
the core untouched.

Does not yet show: consolidation at biographical scale (current capacity
is episodic-scale), semantic key routing on live embeddings, and
integration with a running LLM core. These are the next milestones.
