"""
experiment_3b_complex_overrides.py — Усложнённый сценарий переопределений.

Цепочки изменений: один ключ меняет значение 3-4 раза на протяжении датасета.
6 сценариев, N тем, повторяемость по 3 seed.

Гипотеза: матрёшка (L1→L2 иерархия) сохраняет актуальное значение
для каждого ключа, где плоская память интерферирует.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import numpy as np

D = 64
FACTS = 8          # всего пар на тему (включая переопределения)
FILLER = 4         # шумовых пар
REPLAY_N = 4
SEEDS = [0, 42, 100]
TOPIC_COUNTS = [10, 25, 50, 100]
EPOCHS_TEACH = 30
EPOCHS_REPLAY = 15
LR_TEACH = 0.01
LR_REPLAY = 0.001

# ── 6 сценариев. Каждый имеет ключ и цепочку из 3-4 значений ──
# Формат: (имя_ключа, [значение_1, значение_2, значение_3, значение_4])
# Каждые N_OV тем значение сдвигается на шаг вперёд.
SCENARIOS = [
    ("color",     ["white",   "blue",     "green",    "red"]),
    ("tariff",    ["basic",   "premium",  "enterprise","free"]),
    ("database",  ["pg16",    "pg17",     "mysql8",   "sqlite"]),
    ("framework", ["react",   "vue",      "svelte",   "nextjs"]),
    ("branding",  ["aura",    "aurum",    "forma",    "nucleus"]),
    ("office",    ["moscow",  "spb",      "kazan",    "online"]),
]
N_SCENARIOS = len(SCENARIOS)
OVERRIDES_PER_SCENARIO = 4  # число изменений на сценарий (длина цепочки)


def make_chain_dataset(N, seed):
    """
    Создаёт датасет на N тем с цепочками переопределений.

    Каждый сценарий меняет значение каждые (N // OVERRIDES_PER_SCENARIO) тем,
    не чаще 1 изменения за тему.

    Возвращает:
      all_facts[t]    — list of (key, value) актуальных фактов для темы t
      chain_queries[t] — list of (key, stale_value) для темы t

    Ключи — это случайные фиксированные вектора для каждого сценария.
    Значения — случайные вектора, но разные для разных шагов цепочки.
    """
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    # Фиксируем вектора ключей для всех сценариев
    scenario_keys = []
    scenario_values = []  # список списков: для каждого сценария все 4 вектора значений
    for sname, chain in SCENARIOS:
        key = torch.randn(D) / torch.randn(D).norm()
        scenario_keys.append(key)
        vals = []
        for _ in chain:
            v = torch.randn(D) / torch.randn(D).norm()
            vals.append(v)
        scenario_values.append(vals)

    # Планируем изменения: на каких темах какой сценарий меняется
    # Каждый сценарий меняется (OVERRIDES_PER_SCENARIO - 1) раз (первый — начальное значение)
    changes_per_scenario = OVERRIDES_PER_SCENARIO - 1  # 3 изменения на 4 значения
    change_schedule = {}  # {topic_idx: [(scenario_idx, new_val_idx), ...]}
    for si in range(N_SCENARIOS):
        # Равномерно распределяем изменения по темам
        if N <= 1:
            continue
        change_topics = np.linspace(1, N-1, changes_per_scenario, dtype=int).tolist()
        # Убираем дубликаты
        change_topics = sorted(set(change_topics))
        # Если не хватило мест — добавляем последние темы
        while len(change_topics) < changes_per_scenario:
            t = max(change_topics) + 1
            if t < N:
                change_topics.append(t)
            else:
                change_topics = change_topics[:changes_per_scenario]
                break
        change_topics = change_topics[:changes_per_scenario]

        for vi, topic_idx in enumerate(change_topics):
            # vi+1 — индекс нового значения (1, 2, 3)
            if topic_idx not in change_schedule:
                change_schedule[topic_idx] = []
            change_schedule[topic_idx].append((si, vi + 1))

    # Собираем текущее значение для каждого сценария на каждом шаге
    # current_value_idx[si] = какой индекс от scenario_values[si] сейчас активен
    current_value_idx = [0] * N_SCENARIOS  # начинаем с value[0]

    # Генерируем обычные (не-сценарные) ключи и их значения
    # Для каждой темы: FACTS - (число переопределений в этой теме) обычных фактов
    # Плюс FILLER шумовых

    all_facts = []
    all_chain_queries = []
    all_current = []  # актуальные пары (key, current_value) для check

    for t in range(N):
        # Применяем изменения на этой теме
        if t in change_schedule:
            for si, new_vi in change_schedule[t]:
                current_value_idx[si] = new_vi

        # Текущие значения всех сценариев
        current_pairs = []
        stale_pairs = []
        for si in range(N_SCENARIOS):
            key = scenario_keys[si]
            cur_vi = current_value_idx[si]
            cur_val = scenario_values[si][cur_vi]
            current_pairs.append((key, cur_val))
            # stale = все предыдущие значения этого сценария (кроме текущего)
            for pvi in range(cur_vi):
                stale_pairs.append((key, scenario_values[si][pvi]))

        # Дополнительные обычные факты (не сценарии)
        extra_pairs = []
        for _ in range(FACTS - N_SCENARIOS):
            k = torch.randn(D) / torch.randn(D).norm()
            v = torch.randn(D) / torch.randn(D).norm()
            extra_pairs.append((k, v))

        # Шум
        noise = []
        for _ in range(FILLER):
            k = torch.randn(D) / torch.randn(D).norm()
            v = torch.randn(D) / torch.randn(D).norm()
            noise.append((k, v))

        # Тренировочные пары = сценарии + обычные + шум
        train_pairs = current_pairs + extra_pairs + noise

        all_facts.append(current_pairs)   # актуальные (без шума)
        all_chain_queries.append(stale_pairs)  # устаревшие
        all_current.append(train_pairs)   # всё для обучения

    return all_current, all_facts, all_chain_queries, scenario_keys


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


def train_model(m, pairs, epochs, lr):
    opt = optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        for k, v in pairs:
            opt.zero_grad()
            loss = nn.MSELoss()(m(k.unsqueeze(0)).squeeze(0), v)
            loss.backward()
            opt.step()


def recall_on_pairs(m, pairs):
    """Средний max(0, cosine_similarity) по списку пар."""
    with torch.no_grad():
        sims = [max(0, torch.cosine_similarity(
            m(k.unsqueeze(0)), v.unsqueeze(0)).item()) for k, v in pairs]
        return sum(sims) / len(sims) if sims else 0


def recall_hierarchy(l1, l2, pairs, threshold=0.3):
    """Иерархический recall: L1 → L2."""
    with torch.no_grad():
        sims = []
        for k, v in pairs:
            pred_l1 = l1(k.unsqueeze(0))
            sim_l1 = max(0, torch.cosine_similarity(pred_l1, v.unsqueeze(0)).item())
            if sim_l1 >= threshold:
                sims.append(sim_l1)
            else:
                pred_l2 = l2(k.unsqueeze(0))
                sim_l2 = max(0, torch.cosine_similarity(pred_l2, v.unsqueeze(0)).item())
                sims.append(sim_l2)
        return sum(sims) / len(sims) if sims else 0


def run_seed(N, seed):
    """Один запуск для seed, возвращает метрики для последней темы."""
    all_train, all_current, all_stale, scenario_keys = make_chain_dataset(N, seed)
    all_facts_for_replay = [p[:REPLAY_N] for p in all_current]

    # ── A. Flat + replay ──
    flat_wr = MLP(D, 32)
    for t in range(N):
        if t == 0:
            train_model(flat_wr, all_train[t], EPOCHS_TEACH, LR_TEACH)
        else:
            replay = sum(all_facts_for_replay[:t], [])
            train_model(flat_wr, all_train[t] + replay, EPOCHS_REPLAY, LR_REPLAY)

    # ── B. Матрёшка ──
    l2 = MLP(D, 64)
    for t in range(N):
        l1 = MLP(D, 16)
        train_model(l1, all_train[t], EPOCHS_TEACH, LR_TEACH)
        if t > 0:
            replay = sum(all_facts_for_replay[:t], [])
            train_model(l2, all_train[t] + replay, EPOCHS_REPLAY, LR_REPLAY)
        else:
            train_model(l2, all_train[t], EPOCHS_REPLAY, LR_REPLAY)
        l1_last = l1

    # ── Тест на последней теме ──
    tl = N - 1
    flat_cur = recall_on_pairs(flat_wr, all_current[tl])
    flat_stale = recall_on_pairs(flat_wr, all_stale[tl]) if all_stale[tl] else 0

    mat_cur = recall_hierarchy(l1_last, l2, all_current[tl])
    mat_stale = recall_hierarchy(l1_last, l2, all_stale[tl]) if all_stale[tl] else 0

    # Per-scenario recall for matryoshka (поключевой)
    scenario_recalls = {}
    for si, key in enumerate(scenario_keys):
        # Ищем пару с этим ключом среди current
        for k, v in all_current[tl]:
            if torch.allclose(k, key):
                scenario_recalls[SCENARIOS[si][0]] = recall_hierarchy(l1_last, l2, [(k, v)])
                break

    return (flat_cur, flat_stale, mat_cur, mat_stale, scenario_recalls)


# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════
print("=" * 80)
print("EXPERIMENT 3B: Усложнённый сценарий — цепочки переопределений")
print("  ", N_SCENARIOS, "сценариев x", OVERRIDES_PER_SCENARIO, "изменений каждый")
print("  ", len(SEEDS), "seed'а, темы:", TOPIC_COUNTS)
print("=" * 80)
print()

for N in TOPIC_COUNTS:
    print(f"\n{'#'*60}")
    print(f"#  N = {N} тем")
    print(f"{'#'*60}")

    all_fc = []
    all_fs = []
    all_mc = []
    all_ms = []
    all_sr = []

    for seed in SEEDS:
        fc, fs, mc, ms, sr = run_seed(N, seed)
        all_fc.append(fc)
        all_fs.append(fs)
        all_mc.append(mc)
        all_ms.append(ms)
        all_sr.append(sr)

    fc_mean = np.mean(all_fc)
    fs_mean = np.mean(all_fs)
    mc_mean = np.mean(all_mc)
    ms_mean = np.mean(all_ms)

    print(f"\n  Среднее по {len(SEEDS)} seed'ам (последняя тема #{N-1}):")
    print(f"  {'':>20s}  {'Flat+r':>10s}  {'Матрёшка':>10s}  {'Преимущество':>14s}")
    print(f"  {'─'*56}")
    print(f"  {'Recall (актуальное)':>20s}  {fc_mean:10.4f}  {mc_mean:10.4f}  {mc_mean-fc_mean:+14.4f}")
    print(f"  {'Recall (устаревшее)':>20s}  {fs_mean:10.4f}  {ms_mean:10.4f}  {ms_mean-fs_mean:+14.4f}")
    print(f"  {'Разрыв (акт-уст)':>20s}  {fc_mean-fs_mean:10.4f}  {mc_mean-ms_mean:10.4f}  {(mc_mean-ms_mean)-(fc_mean-fs_mean):+14.4f}")

    # Поключевой recall матрёшки
    print(f"\n  Поключевой recall матрёшки (актуальное, среднее по seed'ам):")
    for sname in [s[0] for s in SCENARIOS]:
        vals = [sr[sname] for sr in all_sr]
        print(f"    {sname:>15s}:  {np.mean(vals):.4f}  (min={np.min(vals):.4f}, max={np.max(vals):.4f})")
    print()

print("\n" + "=" * 80)
print("ИТОГ")
print("=" * 80)
print(f"\nКлючевой вопрос: матрёшка стабильно удерживает актуальное")
print(f"значение по всем сценариям через цепочку из {OVERRIDES_PER_SCENARIO} изменений?")
print(f"Повторяемость по {len(SEEDS)} seed'ам.\n")
