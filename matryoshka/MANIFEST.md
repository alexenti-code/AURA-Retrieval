# Matryoshka Continuity Weights

**Version:** 1.3.1
**Status:** Public conceptual disclosure
**Author:** Alexenti / AURA
**Repository:** alexenti-code/AURA-Retrieval
**Canonical branch:** `docs/matryoshka-public-disclosure-v1`

## Declaration

Matryoshka is an architecture for persistent autobiographical continuity of a language-model core.

It is a self-organizing plastic parameter substrate of autobiographical memory.

It is not a retrieval wrapper, a second reasoning agent, a prompt assembly system, or ordinary continual fine-tuning. Neither is it a governance layer, policy engine, safety guardrail, permission system, tool-access mechanism, executive control layer, or a mechanism for altering the stable core. Its purpose is to give an already capable and ordinarily stable cognitive core the ability to create, preserve and use its own memories of lived events.

[ANATOMY.md](ANATOMY.md) explains the anatomy of the plastic parameter substrate in more detail.

## Core formulation

Let:

\[
K = \text{stable cognitive core}
\]

\[
\Phi(t) = \text{self-authored plastic parameter memory at time }t
\]

The active subject is:

\[
\mathcal{A}(t) = K + \Phi(t)
\]

The core remains stable through normal operation. Experience changes the personal plastic state:

\[
(K,\Phi_t) \xrightarrow{\text{experience}} (K,\Phi_{t+1}).
\]

This change is intended to preserve personal continuity, not to make the core generally more capable.

## Architectural axioms

1. **One semantic subject.** The cognitive core \(K\) alone performs semantic acts: interpretation, attention, reading, writing, evaluation, reasoning, intention and action.
2. **Passive plastic substrate.** \(\Phi\) is passive parameter matter. It has no goals, agency, independent interpretation, planning, retrieval policy or initiative. It only stores numerical state.
3. **Self-authored memory.** The core itself determines whether and how lived experience is recorded in \(\Phi\), and how its own parameter state is read as part of future cognition.
4. **Autobiographical purpose.** The aim is not routine capability training. The aim is memory of people, events, documents, decisions, consequences, obligations, perceptions and temporal context belonging to this particular instance of the core.
5. **Temporal continuity.** A memory trace may encode relationships among entity, event time, observation time, version, source, status and consequence. New information does not erase history by definition; it may form a new temporally related trace.
6. **Nested timescales.** Personal memory may be organized as nested plastic layers from immediate working traces through fresh experience, episodes, projects and biography, while the cognitive core remains the most stable layer.
7. **Experimental separation.** Present implementations may place \(\Phi\) in separate weight files or parameter modules because current infrastructure makes this inspectable and reversible. This separation is an implementation choice, not a claim that memory is another intellect.

## Distinction

Matryoshka distinguishes between:

- pretraining a model to acquire general competence;
- rare, owner-initiated revision of the stable core to add general capabilities — a separate procedure outside Matryoshka; and
- self-authored creation of personal memories in plastic parameters during the operational life of an already capable core.

A model does not become a different intelligence merely because an event is remembered. It becomes the same intelligence with a changed personal history.

## External artifacts

Original documents, signatures, hashes, timestamps and source records remain independent artifacts of reality. They are not competitors to parametric memory. Plastic memory supports continuity of cognition; external artifacts support verification, exactness and evidentiary integrity.

## Research question

Can a stable language-model core preserve and use a specific autobiographical history through self-authored plastic parameter memory, without receiving the whole original history again in a prompt or through retrieval at evaluation time?

## Scope

This document intentionally describes architectural principles and does not disclose implementation-specific mechanisms for writing, locating, updating, protecting or consolidating plastic parameter states.
