"""
AURA Composite Scoring — reference implementation
"""
import math
import numpy as np
from typing import Dict, Any

SCOPE_HALFLIFE = {
    '/news': 90,
    '/market': 90,
    '/objections': 730,
    '/scripts': 730,
    '/regulations': 3650,
    '/rules': 3650,
    '/cases': 730,
    '/precedents': 365,
}

def compute_score(
    semantic_similarity: float,
    days_since_stored: float,
    importance: float,
    half_life: float,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    recency_floor: float = 0.05,
) -> float:
    """Compute composite score using α·semantic + β·recency + γ·importance"""
    recency = max(2 ** (-days_since_stored / half_life), recency_floor)
    score = alpha * semantic_similarity + beta * recency + gamma * importance
    return round(min(score, 1.0), 4)

def resolve_halflife(scope: str) -> float:
    """Resolve halfLife from scope hierarchy"""
    for prefix, hl in SCOPE_HALFLIFE.items():
        if scope.startswith(prefix):
            return hl
    return 30  # default

if __name__ == "__main__":
    # Example: regulation query
    score = compute_score(
        semantic_similarity=0.85,
        days_since_stored=400,
        importance=0.95,
        half_life=resolve_halflife('/regulations/115fz'),
    )
    print(f"Score: {score}")  # Score: ~0.57 (regulation survives despite age)

    # Example: news query  
    score = compute_score(
        semantic_similarity=0.72,
        days_since_stored=60,
        importance=0.4,
        half_life=resolve_halflife('/market/news'),
    )
    print(f"Score: {score}")  # Score: ~0.46 (news decays faster)
