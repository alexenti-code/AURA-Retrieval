"""
experiment_3_dialogue_overrides.py — Диалоговый датасет с отменой/переопределением решений.

Гипотеза: матрёшка выигрывает у плоской памяти в сценарии, где один
и тот же ключ получает разные значения в разное время (переопределение).

Сценарий:
  - N тем по 5 фактов каждая
  - В каждой теме один из фактов — ПЕРЕОПРЕДЕЛЕНИЕ:
    тот же ключ, новое значение (например, "цвет кнопок: белый" → "цвет кнопок: синий")
  - Остальные факты — обычные (каждый ключ один раз)
  - Проверка: после обучения на всех темах, recall по АКТУАЛЬНЫМ значениям
    (должен быть высоким) vs recall по УСТАРЕВШИМ значениям (должен быть низким)

Модели:
  A. Flat:        MLP 64→32→64, sequential + replay (как в статье)
  B. Matryoshka:  L1: 64→16→64, L2: 64→64→64 (как в статье)

Метрики:
  - Recall по актуальным фактам (чем выше, тем лучше)
  - Recall по устаревшим фактам (чем ниже, тем лучше — значит, модель не путается)
  - Override accuracy: доля случаев, где актуальный recall > устаревшего recall
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

torch.manual_seed(42)
D = 64
FACTS_PER_TOPIC = 5   # всего фактов на тему
OVERRIDE_IDX = 0       # какой из 5 фактов переопределяется
FILLER = 3
REPLAY_N = 3
TOPIC_COUNTS = [5, 10, 20, 30, 50]
EPOCHS_TEACH = 30
EPOCHS_REPLAY = 15
LR_TEACH = 0.01
LR_REPLAY = 0.001


def make_override_topics(N):
    """
    Создаёт датасет с переопределениями.

    Возвращает:
      all_facts[t] — список пар (key, value_актуальное) для темы t
      all_override_pairs[t] — список пар (key, value_устаревшее) для темы t
      override_keys[t] — множество ключей, у которых было переопределение

    В каждой теме факт OVERRIDE_IDX имеет:
      - версию А (устаревшую) — подаётся как "старый факт" при обучении L2/later
      - версию Б (актуальную) — подаётся как "новый факт" в этой теме
    """
    all_facts = []            # актуальные пары для темы
    all_override_pairs = []   # устаревшие пары (версия А)
    override_keys = []        # ключ, который был переопределён

    for t in range(N):
        pairs = []
        overrides = []
        ov_keys = []

        for fi in range(FACTS_PER_TOPIC):
            key = torch.randn(D) / torch.randn(D).norm()
            if fi == OVERRIDE_IDX and t > 0:
                # Переопределение: тот же ключ, что в предыдущей теме, новое значение
                old_k, old_v = all_facts[t - 1][OVERRIDE_IDX]
                new_v = torch.randn(D) / torch.randn(D).norm()
                pairs.append((old_k, new_v))    # актуальное (новое)
                overrides.append((old_k, old_v)) # устаревшее (старое)
                ov_keys.append(old_k)
            else:
                val = torch.randn(D) / torch.randn(D).norm()
                pairs.append((key, val))
                overrides.append(None)
                ov_keys.append(None)

        # Шум
        for _ in range(FILLER):
            k = torch.randn(D) / torch.randn(D).norm()
            v = torch.randn(D) / torch.randn(D).norm()
            pairs.append((k, v))

        all_facts.append(pairs[:FACTS_PER_TOPIC])  # без шума — для recall
        all_override_pairs.append([p for p in overrides if p is not None])
        override_keys.append([k for k in ov_keys if k is not None])

    return all_facts, all_override_pairs, override_keys


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


def recall_override(m, facts, override_pairs):
    """
    Два recall:
      current = recall по актуальным фактам
      stale   = recall по устаревшим (переопределённым) фактам
    """
    with torch.no_grad():
        cur = [max(0, torch.cosine_similarity(
            m(k.unsqueeze(0)), v.unsqueeze(0)).item()) for k, v in facts]
        # Устаревшие: только те, что были переопределены
        stale = []
        for k, v in override_pairs:
            sim = max(0, torch.cosine_similarity(
                m(k.unsqueeze(0)), v.unsqueeze(0)).item())
            stale.append(sim)

        avg_cur = sum(cur) / len(cur) if cur else 0
        avg_stale = sum(stale) / len(stale) if stale else 0
        return avg_cur, avg_stale


def train_model(m, pairs, epochs, lr):
    opt = optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        for k, v in pairs:
            opt.zero_grad()
            loss = nn.MSELoss()(m(k.unsqueeze(0)).squeeze(0), v)
            loss.backward()
            opt.step()


def recall_hierarchy(l1, l2, facts, override_pairs, threshold=0.3):
    """
    Иерархический recall: сначала L1 (быстрая), если confidence < threshold — L2.

    Возвращает (avg_current_recall, avg_stale_recall).
    """
    with torch.no_grad():
        cur = []
        for k, v in facts:
            # Пробуем L1
            pred_l1 = l1(k.unsqueeze(0))
            sim_l1 = max(0, torch.cosine_similarity(pred_l1, v.unsqueeze(0)).item())
            if sim_l1 >= threshold:
                cur.append(sim_l1)
            else:
                # Fallback к L2
                pred_l2 = l2(k.unsqueeze(0))
                sim_l2 = max(0, torch.cosine_similarity(pred_l2, v.unsqueeze(0)).item())
                cur.append(sim_l2)

        stale = []
        for k, v in override_pairs:
            # Пробуем L1 — устаревшее значение
            pred_l1 = l1(k.unsqueeze(0))
            sim_l1_stale = max(0, torch.cosine_similarity(pred_l1, v.unsqueeze(0)).item())
            # Пробуем L2 — устаревшее значение
            pred_l2 = l2(k.unsqueeze(0))
            sim_l2_stale = max(0, torch.cosine_similarity(pred_l2, v.unsqueeze(0)).item())
            # Если L1 уверенно возвращает устаревшее — это плохо (не увидел нового)
            # Если L2 возвращает устаревшее — ожидаемо (архив)
            # Берём MAX — worst case для матрёшки
            stale.append(max(sim_l1_stale, sim_l2_stale))

        avg_cur = sum(cur) / len(cur) if cur else 0
        avg_stale = sum(stale) / len(stale) if stale else 0
        return avg_cur, avg_stale


def run_experiment(N):
    """Прогон flat+r vs matryoshka на N темах с переопределениями."""
    all_facts, all_overrides, ov_keys = make_override_topics(N)

    # Формируем тренировочные пары (с шумом)
    all_train_pairs = []
    for t in range(N):
        pairs = []
        for fi in range(FACTS_PER_TOPIC):
            k, v = all_facts[t][fi]
            pairs.append((k, v))
        for _ in range(FILLER):
            k = torch.randn(D) / torch.randn(D).norm()
            v = torch.randn(D) / torch.randn(D).norm()
            pairs.append((k, v))
        all_train_pairs.append(pairs)

    # Все факты для replay (первые REPLAY_N из каждой темы)
    all_facts_for_replay = [f[:REPLAY_N] for f in all_facts]

    # ── A. Flat + replay ──
    flat_wr = MLP(D, 32)
    for t in range(N):
        if t == 0:
            train_model(flat_wr, all_train_pairs[t], EPOCHS_TEACH, LR_TEACH)
        else:
            replay = sum(all_facts_for_replay[:t], [])
            train_model(flat_wr, all_train_pairs[t] + replay, EPOCHS_REPLAY, LR_REPLAY)

    # ── B. Матрёшка ──
    l2 = MLP(D, 64)
    l1_last = None
    for t in range(N):
        l1 = MLP(D, 16)
        train_model(l1, all_train_pairs[t], EPOCHS_TEACH, LR_TEACH)
        if t > 0:
            replay = sum(all_facts_for_replay[:t], [])
            train_model(l2, all_train_pairs[t] + replay, EPOCHS_REPLAY, LR_REPLAY)
        else:
            train_model(l2, all_train_pairs[t], EPOCHS_REPLAY, LR_REPLAY)
        l1_last = l1  # сохраняем L1 последней темы для иерархического recall

    # ── Метрики ──
    # Тестируем ТОЛЬКО последнюю тему (после тренировки на всех предыдущих)
    # Это сценарий: «после длинного диалога ты помнишь текущее решение?»
    t_last = N - 1
    if t_last == 0:
        # N=1 — нет переопределений
        return (0, 0, 0, 1, 0, 0, 0, 1)

    # Flat+r: plain recall
    fc, fs = recall_override(flat_wr, all_facts[t_last], all_overrides[t_last])

    # Матрёшка: иерархический recall (L1 → L2)
    mc, ms = recall_hierarchy(l1_last, l2, all_facts[t_last], all_overrides[t_last])

    return (fc, fs, 1 if fc > fs else 0, 1,
            mc, ms, 1 if mc > ms else 0, 1)


# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════
print("=" * 72)
print("ЭКСПЕРИМЕНТ 3: Диалоговый датасет — переопределение решений")
print("=" * 72)
print(f"\n  d={D}, фактов/тему={FACTS_PER_TOPIC}, шум/тему={FILLER}")
print(f"  факт #{OVERRIDE_IDX+1} в каждой теме (начиная с 1) = переопределение предыдущего")
print(f"  replay: {REPLAY_N} фактов из каждой предыдущей темы\n")

header = (f"{'Темы':>6s}  "
          f"{'Flat+r[cur]':>12s}  {'Flat+r[stale]':>14s}  {'Flat+r OK':>9s}  "
          f"{'Matry[cur]':>11s}  {'Matry[stale]':>13s}  {'Matry OK':>9s}")
print(header)
print("-" * 75)

results = {}
for N in TOPIC_COUNTS:
    r = run_experiment(N)
    results[N] = r
    (fc, fs, fok, fn, mc, ms, mok, mn) = r
    print(f"{N:6d}  {fc:12.4f}  {fs:14.4f}  {fok:3d}/{fn:<3d}      "
          f"{mc:11.4f}  {ms:13.4f}  {mok:3d}/{mn:<3d}")

print(f"\n{'─' * 75}")
print(f"\nСводка: recall по актуальным (выше = лучше) и устаревшим (ниже = лучше)\n")
print(f"{'Темы':>6s}  {'Flat+r акт':>10s}  {'Flat+r уст':>11s}  "
      f"{'Матрёшка акт':>13s}  {'Матрёшка уст':>14s}  {'Перевес матр':>13s}")
print("-" * 70)
for N in TOPIC_COUNTS:
    fc, fs, fok, fn, mc, ms, mok, mn = results[N]
    override_gap = mc - fc
    stale_gap = fs - ms  # положительный = матрёшка лучше давит устаревшее
    print(f"{N:6d}  {fc:10.4f}  {fs:11.4f}  {mc:13.4f}  {ms:14.4f}  "
          f"{override_gap:+13.4f}")

print(f"\nКлючевой вопрос: у кого лучше отделение актуального от устаревшего?")
print(f"(актуальный recall - устаревший recall, чем больше разрыв, тем лучше)\n")
print(f"{'Темы':>6s}  {'Flat+r разрыв':>13s}  {'Матрёшка разрыв':>16s}  {'Преимущество':>13s}")
print("-" * 50)
for N in TOPIC_COUNTS:
    fc, fs, fok, fn, mc, ms, mok, mn = results[N]
    flat_gap = fc - fs
    matry_gap = mc - ms
    advantage = matry_gap - flat_gap
    print(f"{N:6d}  {flat_gap:13.4f}  {matry_gap:16.4f}  {advantage:+13.4f}")
