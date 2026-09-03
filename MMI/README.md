# MMI

**MMI** is the external memory interface of the Matryoshka anatomy in its present experimental form.

It is not a second reasoning agent, not a retrieval wrapper, not a governance layer, and not an executive control system.

In the current stage, MMI may exist as a minimal prompt adapter placed before each model call. Its role is to inform the stable core that a separate autobiographical plastic memory exists and can be used.

## Role

MMI provides architectural separation and compatibility of access between:

- **K** — the stable language-model core;
- **Φ(t)** — the separate plastic autobiographical parameter body.

MMI does not interpret experience, prescribe memory content, assign goals, or decide behaviour.

## Present form

The present experimental form may include:

- a prompt prelude injected before a request;
- a minimal memory anatomy declaration;
- a compatibility schema for layers and allowed operations;
- a runtime adapter that assembles these parts for a model call.

## Future form

The same principle may later become a built-in architectural interface inside a transformer.

This directory contains only a minimal v0 skeleton.
