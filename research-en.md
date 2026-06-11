# AURA: An Externalized, State-Separated Multi-Agent Architecture for Mitigation of Autophagy in Long-Term LLM Context-Management

**Authors:** ООО «АУРУМ ЭСТЕЙТ» / AURA.KIM  
**Category:** Computer Science — Software Engineering (cs.SE); Artificial Intelligence (cs.AI)  
**License:** Apache License 2.0  
**Date:** June 2026

## Abstract
Modern Large Language Models (LLMs) present a fundamental computational bottleneck during continuous execution due to linear O(N) degradation of decoding latency and semantic decay within extended context windows. Attempts to achieve online self-training via parametric model updating typically trigger autophagy—an exponential entropy loop where the model degrades by training on its own generative noise.

This paper presents AURA (Advanced Unified Retrieval Architecture), a non-parametric, state-separated architectural pattern that externalizes episodic and long-term memory into an independent software runtime. By implementing an asynchronous hierarchy of six specialized deterministic agent roles grouped into orthogonal control loops, AURA isolates the computation core (the LLM) from the state repository. We demonstrate that discrete quantization of agent layers (3 → 6 → 9) yields optimal convergence limits, providing deterministic guardrails against regression without requiring dynamic weight adaptation of the underlying neural models.
