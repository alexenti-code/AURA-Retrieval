"""
experiment_mlp.py — ключевой эксперимент: иерархия памяти с MLP.

Три условия:
  A. Flat без replay: один MLP учится последовательно на всех темах
  B. Flat + replay: тот же MLP, но на каждой новой теме перемешивает
     новые данные с REPLAY_N фактами из каждой предыдущей темы
  C. Матрёшка: L1 (64->16->64) учит тему и сбрасывается, L2 (64->64->64)
     копит знания через replay (как B)

Результат (seed=42, 50 тем, 650 пар):
                    avg     recall>0.5
  Flat без replay:  0.00    1/50
  Flat + replay:    0.48    25/50
  Матрёшка:         0.66    47/50
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

torch.manual_seed(42)
D = 64
N = 50
FACTS = 5
FILLER = 8
REPLAY_N = 4


def topic_pairs(tid, n):
    return [(torch.randn(D)/torch.randn(D).norm(), torch.randn(D)/torch.randn(D).norm()) for _ in range(n)]


class MLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.net = nn.Sequential(OrderedDict([
            ("l1", nn.Linear(d, h)),
            ("act", nn.ReLU()),
            ("l2", nn.Linear(h, d)),
        ]))
    def forward(self, x): return self.net(x)


def recall(m, facts):
    """Средний max(0, cosine_similarity) — клип в 0, т.к. отрицательные
    значения означают «память отсутствует» (случайный шум)."""
    with torch.no_grad():
        sims = [max(0, torch.cosine_similarity(
            m(k.unsqueeze(0)), v.unsqueeze(0)).item()) for k, v in facts]
        return sum(sims) / len(sims)


def train(m, pairs, epochs, lr):
    opt = optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        for k, v in pairs:
            opt.zero_grad()
            loss = nn.MSELoss()(m(k.unsqueeze(0)).squeeze(0), v)
            loss.backward()
            opt.step()


all_facts = [topic_pairs(t, FACTS) for t in range(N)]
all_filler = [topic_pairs(t, FILLER) for t in range(N)]
all_pairs = [all_facts[t] + all_filler[t] for t in range(N)]


# ── A. Flat без replay ──
print("=" * 60)
print("ЭКСПЕРИМЕНТ: flat vs flat+replay vs matryoshka")
print("=" * 60)
print(f"\n  d={D}, {N} тем, {FACTS}+{FILLER} пар/тему")
print(f"  replay: {REPLAY_N} фактов из каждой предыдущей темы")
print(f"  обучение: 30 эпох/тему (lr=0.01), L2: 15 эпох (lr=0.001)\n")

flat_nr = MLP(D, 32)
for t in range(N):
    train(flat_nr, all_pairs[t], 30, 0.01)

# ── B. Flat + replay ──
flat_wr = MLP(D, 32)
for t in range(N):
    if t == 0:
        train(flat_wr, all_pairs[t], 30, 0.01)
    else:
        replay = sum([all_facts[pt][:REPLAY_N] for pt in range(t)], [])
        train(flat_wr, all_pairs[t] + replay, 15, 0.001)

# ── C. Матрёшка ──
l2 = MLP(D, 64)
for t in range(N):
    l1 = MLP(D, 16)
    train(l1, all_pairs[t], 30, 0.01)
    replay = sum([all_facts[pt][:REPLAY_N] for pt in range(t)], [])
    train(l2, all_pairs[t] + replay, 15, 0.001)

# ── Результаты ──
print(f"{'Тема':>5s}  {'Flat (без replay)':>18s}  {'Flat + replay':>14s}  {'Матрёшка':>10s}")
print("-" * 55)
for t in range(N):
    fnr = recall(flat_nr, all_facts[t])
    fwr = recall(flat_wr, all_facts[t])
    mr = recall(l2, all_facts[t])
    print(f"{t:5d}  {fnr:.4f}             {fwr:.4f}          {mr:.4f}")

print(f"\n{'─' * 55}")
fnr_ok = sum(1 for t in range(N) if recall(flat_nr, all_facts[t]) > 0.5)
fwr_ok = sum(1 for t in range(N) if recall(flat_wr, all_facts[t]) > 0.5)
l2_ok = sum(1 for t in range(N) if recall(l2, all_facts[t]) > 0.5)
avgs = [sum(recall(m, all_facts[t]) for t in range(N)) / N
        for m in (flat_nr, flat_wr, l2)]

print(f"среднее:       {avgs[0]:.4f}             {avgs[1]:.4f}          {avgs[2]:.4f}")
print(f"recall > 0.5:  {fnr_ok}/{N}              {fwr_ok}/{N}           {l2_ok}/{N}")

for tidx in [0, 10, 20, 30, 40, 49]:
    fnr_s = recall(flat_nr, all_facts[tidx])
    fwr_s = recall(flat_wr, all_facts[tidx])
    l2_s = recall(l2, all_facts[tidx])
    print(f"тема {tidx:2d}:       {fnr_s:.4f}             {fwr_s:.4f}          {l2_s:.4f}")

print(f"\nВыводы:")
print(f"  Replay лечит catastrophic forgetting (flat: {avgs[0]:.2f} → flat+r: {avgs[1]:.2f})")
print(f"  Матрёшка выигрывает у flat+replay на {avgs[2]-avgs[1]:+.3f} в среднем.")
print(f"  Матрёшка сохранила {l2_ok-fwr_ok} тем больше (recall>0.5: {l2_ok} vs {fwr_ok}).")
