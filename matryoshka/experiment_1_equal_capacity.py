"""
experiment_1_equal_capacity.py — Equal Capacity Baseline (MLP).

Проверка: сохраняется ли выигрыш матрёшки при одинаковой ёмкости?

Модели:
  A. Flat-64:        MLP 64→64→64, последовательно, без повтора
  B. Flat-64+replay: MLP 64→64→64, последовательно + replay 4 фактов/тему
  C. Matryoshka-64:  L1: 64→64→64, L2: 64→64→64 (L1 сбрасывается, L2 копит)

Датасет: синтетические ортогональные ключи, 5 фактов + 8 шумовых на тему.
Scaling: 10, 25, 50, 100, 200 тем.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

torch.manual_seed(42)
D = 64
H = 64          # равная ёмкость скрытого слоя
FACTS = 5
FILLER = 8
REPLAY_N = 4
TOPIC_COUNTS = [10, 25, 50, 100, 200]
EPOCHS_TEACH = 30
EPOCHS_REPLAY = 15
LR_TEACH = 0.01
LR_REPLAY = 0.001


def topic_pairs(tid, n):
    """n ортогональных пар ключ→значение для темы tid."""
    return [(torch.randn(D) / torch.randn(D).norm(),
             torch.randn(D) / torch.randn(D).norm()) for _ in range(n)]


class MLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.net = nn.Sequential(OrderedDict([
            ("l1", nn.Linear(d, h)),
            ("act", nn.ReLU()),
            ("l2", nn.Linear(h, d)),
        ]))

    def forward(self, x):
        return self.net(x)


def recall(m, facts):
    """Средний max(0, cosine_similarity) по списку фактов."""
    with torch.no_grad():
        sims = [max(0, torch.cosine_similarity(
            m(k.unsqueeze(0)), v.unsqueeze(0)).item()) for k, v in facts]
        return sum(sims) / len(sims) if sims else 0.0


def train_model(m, pairs, epochs, lr):
    opt = optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        for k, v in pairs:
            opt.zero_grad()
            loss = nn.MSELoss()(m(k.unsqueeze(0)).squeeze(0), v)
            loss.backward()
            opt.step()


def run_experiment(N):
    """Полный прогон трёх моделей на N темах. Возвращает (avg_recalls, ok_counts)."""
    all_facts = [topic_pairs(t, FACTS) for t in range(N)]
    all_filler = [topic_pairs(t, FILLER) for t in range(N)]
    all_pairs = [all_facts[t] + all_filler[t] for t in range(N)]

    # ── A. Flat-64 без replay ──
    flat_nr = MLP(D, H)
    for t in range(N):
        train_model(flat_nr, all_pairs[t], EPOCHS_TEACH, LR_TEACH)

    # ── B. Flat-64 + replay ──
    flat_wr = MLP(D, H)
    for t in range(N):
        if t == 0:
            train_model(flat_wr, all_pairs[t], EPOCHS_TEACH, LR_TEACH)
        else:
            replay = sum([all_facts[pt][:REPLAY_N] for pt in range(t)], [])
            train_model(flat_wr, all_pairs[t] + replay, EPOCHS_REPLAY, LR_REPLAY)

    # ── C. Matryoshka-64 (L1 сбрасывается, L2 копит) ──
    l2 = MLP(D, H)
    for t in range(N):
        l1 = MLP(D, H)
        train_model(l1, all_pairs[t], EPOCHS_TEACH, LR_TEACH)
        if t > 0:
            replay = sum([all_facts[pt][:REPLAY_N] for pt in range(t)], [])
            train_model(l2, all_pairs[t] + replay, EPOCHS_REPLAY, LR_REPLAY)
        else:
            train_model(l2, all_pairs[t], EPOCHS_REPLAY, LR_REPLAY)

    # ── Метрики ──
    avg_nr = sum(recall(flat_nr, all_facts[t]) for t in range(N)) / N
    avg_wr = sum(recall(flat_wr, all_facts[t]) for t in range(N)) / N
    avg_l2 = sum(recall(l2, all_facts[t]) for t in range(N)) / N

    ok_nr = sum(1 for t in range(N) if recall(flat_nr, all_facts[t]) > 0.5)
    ok_wr = sum(1 for t in range(N) if recall(flat_wr, all_facts[t]) > 0.5)
    ok_l2 = sum(1 for t in range(N) if recall(l2, all_facts[t]) > 0.5)

    return (avg_nr, avg_wr, avg_l2), (ok_nr, ok_wr, ok_l2)


# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════
print("=" * 70)
print("ЭКСПЕРИМЕНТ 1: Equal Capacity Baseline (MLP 64→64→64)")
print("=" * 70)
print(f"\n  d={D}, h={H}, фактов/тему={FACTS}, шум/тему={FILLER}")
print(f"  replay: {REPLAY_N} фактов из каждой предыдущей темы")
print(f"  обучение: {EPOCHS_TEACH} эпох/тему (lr={LR_TEACH}), "
      f"replay: {EPOCHS_REPLAY} эпох (lr={LR_REPLAY})\n")

print(f"{'Темы':>6s}  {'Flat-64':>8s}  {'Flat-64+r':>10s}  {'Matryoshka':>11s}  "
      f"{'Flat-64>0.5':>10s}  {'Flat+r>0.5':>10s}  {'Matryosh>0.5':>12s}")
print("-" * 70)

results = {}
for N in TOPIC_COUNTS:
    avgs, oks = run_experiment(N)
    results[N] = (avgs, oks)
    print(f"{N:6d}  {avgs[0]:8.4f}  {avgs[1]:10.4f}  {avgs[2]:11.4f}  "
          f"{oks[0]:3d}/{N:3d}      {oks[1]:3d}/{N:3d}      {oks[2]:3d}/{N:3d}")

print(f"\n{'─' * 70}")
print(f"\nСводка (Equal Capacity MLP 64→64→64)\n")
print(f"{'Темы':>6s}  {'Flat-64':>8s}  {'Flat-64+r':>10s}  {'Matryoshka':>11s}")
print("-" * 40)
for N in TOPIC_COUNTS:
    avgs, _ = results[N]
    print(f"{N:6d}  {avgs[0]:8.4f}  {avgs[1]:10.4f}  {avgs[2]:11.4f}")

print(f"\nRecall > 0.5:\n")
print(f"{'Темы':>6s}  {'Flat-64':>9s}  {'Flat-64+r':>10s}  {'Matryoshka':>11s}")
print("-" * 40)
for N in TOPIC_COUNTS:
    _, oks = results[N]
    print(f"{N:6d}  {oks[0]:3d}/{N:<3d}       {oks[1]:3d}/{N:<3d}       {oks[2]:3d}/{N:<3d}")
