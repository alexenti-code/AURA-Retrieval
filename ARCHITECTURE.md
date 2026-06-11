# AURA Architecture Diagram

## Three-Circuit Execution Model

[User] ── Request ──► ┌──────────────────────────┐
                       │  Hot Circuit (Online)     │ ──► Response (~200ms)
                       │  Colleague (RAG/Executor) │
                       └──────────────────────────┘
                                   │
                             Test logs
                                   ▼
                       ┌──────────────────────────┐
                       │  Warm Circuit (Near-line) │ ◄── Test gen (Intern)
                       │  Teacher (Validator)      │ ──► Weight correction
                       └──────────────────────────┘
                                   │
                          Metrics < 0.80
                                   ▼
                       ┌──────────────────────────┐
                       │  Cold Circuit (Offline)   │
                       │  Executor (Code patcher)  │
                       │       │                   │
                       │       ▼                   │
                       │  Mentor (Guardrail)       │ ──► Validation & deploy
                       └──────────────────────────┘

## Composite Scoring
Score = α·semantic + β·recency + γ·importance

α = 0.5 (semantic weight)
β = 0.3 (recency weight)  
γ = 0.2 (importance weight)

## 6 Cognitive Roles
1. Colleague   - Frontend RAG, answers user queries
2. Intern      - Synthetic generator, edge case explorer
3. Teacher     - Memory validator, weight adjuster
4. Executor    - Code patcher, meta-config optimizer
5. Researcher  - Telemetry auditor, degradation detector
6. Mentor      - Final guardrail, human-in-the-loop
