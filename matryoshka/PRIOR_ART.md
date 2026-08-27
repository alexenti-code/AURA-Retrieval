# Prior Art and Distinctions

## Purpose

Matryoshka does not claim to have invented neural memory, parameter updates, test-time learning, long-context processing, adapters, fast weights, retrieval or external archives.

This document identifies related research and defines the narrower architectural position of Matryoshka.

## Related directions

### Test-Time Training and Titans

Test-Time Training and Titans investigate updating a neural memory module or selected parameters during inference, primarily to process long sequences and retain distant context.

Matryoshka shares the proposition that experience may be represented in plastic parameters. Its stated purpose differs: the plastic substrate is defined as self-authored autobiographical continuity for a stable cognitive core, rather than primarily as a long-context or benchmark optimization mechanism.

### Fast weights

Fast-weight approaches use dynamic parameters or states that change as a function of recent input.

Matryoshka extends the conceptual target from short-lived adaptation to nested timescales of personal memory: immediate experience, episodes, projects, biography and meta-memory.

### Linear associative memory and fast-weight programmers

The direct mechanistic ancestors of the present experimental materials are
classical associative memory models:

- Anderson (1972), "A simple neural network generating an interactive memory"
  and Kohonen (1972), "Correlation matrix memories" — the linear associator:
  storage as a sum of outer products, recall as W·k, with classical
  capacity and interference limits from superposition.
- Schmidhuber (1992), "Learning to Control Fast-Weight Memories: An
  Introduction to Programmable Neural Networks" — fast-weight programmers:
  a slow network writes the weights of a fast network.
- Their modern continuation: linear transformers (Katharopoulos et al.,
  2020), DeltaNet (Schlag et al., 2021), test-time training layers
  (Sun et al., 2024), Titans and MIRAS (Behrouz et al., 2024–2025).

Matryoshka does not claim a new write or read operator. Its narrow
contribution, as defined at the end of this document, is architectural
position: self-authored, passive, autobiographical, temporally versioned
parameter memory belonging to a stable cognitive core.

### PEFT, LoRA and continual adapters

Parameter-efficient methods preserve a foundation-model backbone while training a smaller set of additional parameters. Continual-learning methods often focus on new tasks, domains and skills while reducing catastrophic forgetting.

Matryoshka treats plastic parameter files as a provisional physical form of an instance's own memory, rather than solely as task-specific adaptation modules.

### RAG and external agent memory

RAG, vector stores, event logs, knowledge graphs and agent-memory systems preserve information outside model parameters and present selected material back to the model.

Matryoshka does not deny the role of external archives. It distinguishes them from the model's internal continuity substrate: the original artifact may remain external while the lived and interpreted trace is stored in the plastic parameter state.

### Persistent-identity agents

Persistent-agent designs aim to preserve persona and continuity across sessions, usually through external files, summaries, memory stores and retrieval.

Matryoshka shares the continuity goal, but defines the personal memory substrate as self-authored plastic parameters associated with a stable cognitive core.

## Narrow contribution

The Matryoshka proposition is the combination of:

1. a generally stable language-model core;
2. a passive, mutable parameter substrate belonging to the same instance;
3. semantic control of memory creation and use retained by the core;
4. autobiographical rather than ordinary task-adaptation purpose;
5. temporally and version-aware personal continuity;
6. nested plastic timescales; and
7. a present-day experimental separation of the plastic substrate into inspectable weight files, with a prospective future unification into one architecture.
