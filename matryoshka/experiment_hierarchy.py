"""
experiment_hierarchy.py — ключевой эксперимент статьи.

Сравнивает иерархическую память (матрёшку) с одной плоской памятью
на сценарии: длинный диалог из 4 тем, в начале факт «кнопки белые»,
потом три другие темы, потом возврат к исходной теме.

Гипотеза:
- Плоская память: факты из темы 1 погребены под 120+ парами из тем 2-4
- Матрёшка: факты темы 1 сохранены на L2 при переключении темы

Запуск: python3 experiment_hierarchy.py
"""

import sys
import torch
from core import FastWeightL0, FastWeightL1, FastWeightL2, random_pair

D_MODEL = 128
TOPIC_NAMES = ["UI (кнопки)", "Цены", "Архитектура БД", "Сроки"]
FACTS_PER_TOPIC = 5
FILLER_PER_TOPIC = 10

torch.manual_seed(42)

def topic_facts(topic_id, n):
    facts = []
    for i in range(n):
        torch.manual_seed(topic_id * 100 + i)
        k = torch.randn(D_MODEL)
        v = torch.randn(D_MODEL)
        facts.append((k / k.norm(), v / v.norm()))
    return facts

def filler_pairs(topic_id, n):
    pairs = []
    for i in range(n):
        torch.manual_seed(topic_id * 200 + i)
        k = torch.randn(D_MODEL)
        v = torch.randn(D_MODEL)
        pairs.append((k / k.norm(), v / v.norm()))
    return pairs

def recall_for(memory, facts):
    sims = []
    for k, v_orig in facts:
        v_pred = memory.read(k)
        sim = torch.cosine_similarity(v_pred.unsqueeze(0), v_orig.unsqueeze(0)).item()
        sims.append(sim)
    return sims


# ── 1. Данные ──────────────────────────────────────────────────

all_facts = []
all_filler = []
for t in range(4):
    all_facts.append(topic_facts(t, FACTS_PER_TOPIC))
    all_filler.append(filler_pairs(t, FILLER_PER_TOPIC))


# ── 2. Плоская память (один L1, rank=16, всё в одну матрицу) ──

print("=" * 60)
print("ЭКСПЕРИМЕНТ: иерархия vs плоская память")
print("=" * 60)
print(f"\nДиалог: 4 темы, {FACTS_PER_TOPIC} ключевых фактов + {FILLER_PER_TOPIC} шумовых пар в каждой")
print(f"Всего пар на L1/матрёшку: {4 * (FACTS_PER_TOPIC + FILLER_PER_TOPIC)}\n")

flat = FastWeightL1(D_MODEL)
flat_name = "Плоская (rank≈16)"

for t in range(4):
    for k, v in all_facts[t] + all_filler[t]:
        flat.write(k, v)

flat_recalls = recall_for(flat, all_facts[0])
flat_avg = sum(flat_recalls) / len(flat_recalls)
flat_above = sum(1 for s in flat_recalls if s >= 0.7)
print(f"  {flat_name:30s} тема 0 recall avg={flat_avg:.3f}  выше 0.7: {flat_above}/{len(flat_recalls)}")


# ── 3. Матрёшка (L0 + L1 + L2) ────────────────────────────────

matryoshka_name = "Матрёшка (L0+L1→L2)"

l0 = FastWeightL0(D_MODEL)
l1 = FastWeightL1(D_MODEL)
l2 = FastWeightL2(D_MODEL)

for t in range(4):
    # Записываем факты темы
    for k, v in all_facts[t] + all_filler[t]:
        # Старая пара с L0 → на L1
        if l0._stored_key is not None:
            l1.write(l0._stored_key, l0._stored_val)
        # Новая пара на L0
        l0.write(k, v)

    # Смена темы: L1 → L2, очистка L1
    l2.absorb(l1, TOPIC_NAMES[t])

m_l2_recalls = recall_for(l2, all_facts[0])
m_l2_avg = sum(m_l2_recalls) / len(m_l2_recalls)
m_l2_above = sum(1 for s in m_l2_recalls if s >= 0.7)

m_l1_recalls = recall_for(l1, all_facts[0])
m_l1_avg = sum(m_l1_recalls) / len(m_l1_recalls) if m_l1_recalls else 0

print(f"  {matryoshka_name:30s} тема 0 recall на L2 avg={m_l2_avg:.3f}  выше 0.7: {m_l2_above}/{len(m_l2_recalls)}")
print(f"  {'':30s} тема 0 recall на L1 avg={m_l1_avg:.3f} (ожидается ~0 — тема сброшена)")


# ── 4. Результат ──────────────────────────────────────────────

print(f"\n{'—'*60}")
if m_l2_above > flat_above:
    print(f"РЕЗУЛЬТАТ: Матрёшка сохранила {m_l2_above}/{FACTS_PER_TOPIC} фактов темы 1,")
    print(f"           Плоская память — {flat_above}/{FACTS_PER_TOPIC}.")
    print("           Иерархия защищает старые факты от погребения новыми.")
else:
    diff = flat_avg - m_l2_avg
    print(f"РЕЗУЛЬТАТ: Разница {diff:+.3f} в пользу {'плоской' if diff > 0 else 'матрёшки'}.")
    print("           Для d_model=128 и 4 тем разница может быть мала.")

# Детали по всем темам
print(f"\n{'—'*60}")
print("Детализация по всем темам:")
print(f"{'Тема':20s} {'Плоская (avg)':16s} {'Матрёшка L2 (avg)':20s}")
print(f"{'—'*56}")
for t in range(4):
    flat_s = sum(recall_for(flat, all_facts[t])) / FACTS_PER_TOPIC
    mat_s = sum(recall_for(l2, all_facts[t])) / FACTS_PER_TOPIC
    print(f"{TOPIC_NAMES[t]:20s} {flat_s:.4f}           {mat_s:.4f}")
