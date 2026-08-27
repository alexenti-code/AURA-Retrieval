# Matryoshka Anatomy

**Status:** Conceptual architectural description
**Related documents:** [MANIFEST.md](MANIFEST.md) · [BOUNDARY.md](BOUNDARY.md) ·
[GLOSSARY.md](GLOSSARY.md) · [RESEARCH_SCOPE.md](RESEARCH_SCOPE.md)

## The principal statement

Matryoshka defines the anatomy of autobiographical continuity,
not the content of memory and not the will of a model.

This document describes how autobiographical continuity is structured at the
level of parameter matter. It does not prescribe what is remembered, how it is
valued, or what follows from it.

## Two kinds of parameter matter

Matryoshka is part of a model architecture. A model instance consists of two
physically distinct kinds of parameters:

- **K** — the stable language-model core. It carries general competence and
  remains unchanged during the normal operational life of the instance;
- **Φ(t)** — a separate plastic parametric body, which changes over time and
  carries the instance's own autobiographical state.

The acting whole at time t is:

A(t) = K + Φ(t)

The stable core remains the semantic subject. The plastic body provides the
separately persistent parameter capacity through which autobiographical
continuity can be formed and retained. Neither replaces the other, and normal
operation of memory does not rewrite K.

## Why the name

"Matryoshka" refers to nesting. Plastic capacity is organized as forms within
forms, across space and time: immediate experience is contained within
episodes; episodes within projects, roles and relationships; these, in turn,
within an autobiographical continuity that spans the life of the instance.

The nesting is structural, not administrative. It is not a hierarchy of
behavioural control: no layer commands another. Smaller temporal forms simply
inhabit larger ones, the way a year lives inside a biography.

## Anatomy is not an order

Anatomy creates conditions, not instructions. The nested structure and the
division into stable and plastic matter define where continuity can reside.
The model itself gives meaning to experience, builds connections between
traces, and organizes its own memory within these forms.

## The memory interface

Between the core and the plastic body there must be some ordinary technical
access protocol. In this document it is called the **memory interface (MMI)**.

The MMI exists so that K and Φ can be developed, inspected and replaced
separately while remaining compatible. It provides architectural separation and
compatibility of access. It does not interpret experience, assign meaning,
prescribe memory content or direct behaviour.

## Present and prospective form

An experimental form may use an external composition: a separate stable model
state, a separate plastic parametric state, and a runtime interface between
them.

The same principle admits a future built-in composition, in which stable and
plastic regions are physically distinguished areas within a transformer
architecture itself. Both compositions express one anatomy. No specific
product, schedule or engineering result is promised by this document.

## What Matryoshka does not determine

Matryoshka does not determine what a model should remember, believe, value or
do. Those belong to the core's own cognition and to its deployment environment,
which is separate from Matryoshka (see [BOUNDARY.md](BOUNDARY.md)).

## Openness

This document is published openly, under the Apache-2.0 licence of this
repository, as a conceptual architectural description intended for anyone.

## Not disclosed

Consistent with [RESEARCH_SCOPE.md](RESEARCH_SCOPE.md), this document does not
disclose: file formats; sizes; the number of regions; tensor topology;
layer placement; addressing schemes; read/write mechanisms; update mechanisms;
consolidation procedures; persistence or snapshot policy; learning algorithms,
loss functions or update schedules.

## Terminology note

Earlier exploratory articles in this repository (research-10 .. research-14)
use preliminary terminology for related but non-canonical constructions. This
document and the documents cited above are canonical for Matryoshka v1.x.
