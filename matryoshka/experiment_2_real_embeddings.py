"""
experiment_2_real_embeddings.py — Реальные эмбеддинги.

Проверка: сохраняется ли преимущество матрёшки при реалистичной
интерференции ключей (семантически близкие эмбеддинги).

Модели (как в статье):
  A. Flat:        MLP 64→32→64, sequential
  B. Flat+replay: MLP 64→32→64, sequential + replay
  C. Matryoshka:  L1: 64→16→64, L2: 64→64→64

Датасет: псевдодиалоговые утверждения, эмбеддинги через sentence-transformers.
Scaling: 10, 50, 100 тем.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import numpy as np

torch.manual_seed(42)
D = 64
FACTS = 5
FILLER = 8
REPLAY_N = 4
TOPIC_COUNTS = [10, 50, 100]
EPOCHS_TEACH = 30
EPOCHS_REPLAY = 15
LR_TEACH = 0.01
LR_REPLAY = 0.001

# ── Шаблоны утверждений для тем ──
# Каждая тема: 5 фактов (уникальных для темы) + 8 общих шумовых фраз
TOPIC_TEMPLATES = [
    # Тема 0 — интерфейс
    {"facts": [
        "Цвет кнопок — белые на светлом фоне",
        "Расположение меню — левый верхний угол",
        "Шрифт заголовков — 14pt Inter",
        "Анимация переходов — 300 мс ease-out",
        "Отступы между карточками — 24px",
    ]},
    # Тема 1 — цены
    {"facts": [
        "Базовая цена подписки — 990 рублей в месяц",
        "Скидка при годовой оплате — 20%",
        "Тариф Премиум — 2990 рублей",
        "Бесплатный период — 14 дней",
        "Комиссия за транзакцию — 2.9%",
    ]},
    # Тема 2 — база данных
    {"facts": [
        "Основная СУБД — PostgreSQL 16",
        "Репликация — streaming async, 3 ноды",
        "Резервное копирование — ежедневно в 03:00",
        "Шардирование — по user_id хеш",
        "Миграции — через Alembic, версионные",
    ]},
    # Тема 3 — архитектура
    {"facts": [
        "Микросервисы — 7 сервисов на FastAPI",
        "Межсервисная связь — RabbitMQ + gRPC",
        "Аутентификация — JWT + OAuth2",
        "Кеширование — Redis Cluster, 5 нод",
        "Мониторинг — Prometheus + Grafana",
    ]},
    # Тема 4 — фронтенд
    {"facts": [
        "Фреймворк — React 18 + TypeScript",
        "Стейт-менеджмент — Zustand",
        "Сборка — Vite 5",
        "Тестирование — Vitest + Playwright",
        "Дизайн-система — Radix UI + Tailwind",
    ]},
]

NOISE_SENTENCES = [
    "Погода сегодня солнечная",
    "Вчера был дождь",
    "Кофеварка сломалась",
    "Нужно купить молоко",
    "Пробки на МКАД 8 баллов",
    "Завтра встреча в 10 утра",
    "Отопление отключили",
    "Курс доллара 85 рублей",
    "Соседи делают ремонт",
    "Новый ресторан открылся",
    "Поезд задерживается на 20 минут",
    "Акция в супермаркете до пятницы",
    "Фильм получил Оскар",
    "Температура на улице +22",
    "Абонемент в спортзал закончился",
    "Нужно забрать посылку на почте",
]


def get_embedder():
    """Загружаем nomic-embed-text через sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    emb_dim = model.get_sentence_embedding_dimension()
    print(f"  Размерность эмбеддингов: {emb_dim}")
    return model, emb_dim


def embed_texts(embedder, texts):
    """Превращает список строк в тензор эмбеддингов."""
    embs = embedder.encode(texts, normalize_embeddings=True)
    return torch.tensor(embs, dtype=torch.float32)


def make_topics(embedder, emb_dim, num_topics):
    """Создаёт num_topics тем с последующей проекцией в D=64."""
    all_facts = []
    all_filler = []
    # Проекция 768→64
    projector = nn.Linear(emb_dim, D)
    with torch.no_grad():
        for name, p in projector.named_parameters():
            nn.init.normal_(p, std=0.02)

    for t in range(num_topics):
        if t < len(TOPIC_TEMPLATES):
            fact_texts = TOPIC_TEMPLATES[t]["facts"]
        else:
            base = TOPIC_TEMPLATES[t % len(TOPIC_TEMPLATES)]["facts"]
            fact_texts = [f"{b} (вариант {t // len(TOPIC_TEMPLATES) + 1})" for b in base]

        np.random.seed(t)
        filler_texts = list(np.random.choice(NOISE_SENTENCES, size=FILLER, replace=False))

        all_fact_embs = embed_texts(embedder, fact_texts)
        all_filler_embs = embed_texts(embedder, filler_texts)

        all_fact_proj = projector(all_fact_embs).detach()
        all_filler_proj = projector(all_filler_embs).detach()

        # Нормализуем для стабильности
        all_fact_proj = all_fact_proj / all_fact_proj.norm(dim=1, keepdim=True)
        all_filler_proj = all_filler_proj / all_filler_proj.norm(dim=1, keepdim=True)

        # Пары ключ→значение (ключ = значение для упрощения)
        facts = [(all_fact_proj[i], all_fact_proj[i]) for i in range(FACTS)]
        filler = [(all_filler_proj[i], all_filler_proj[i]) for i in range(FILLER)]
        all_facts.append(facts)
        all_filler.append(filler)

    return all_facts, all_filler


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


def run_experiment(all_facts, all_filler, N):
    """Прогон трёх моделей на N темах с реальными эмбеддингами."""
    all_pairs = [all_facts[t] + all_filler[t] for t in range(N)]

    # ── A. Flat ──
    flat_nr = MLP(D, 32)
    for t in range(N):
        train_model(flat_nr, all_pairs[t], EPOCHS_TEACH, LR_TEACH)

    # ── B. Flat + replay ──
    flat_wr = MLP(D, 32)
    for t in range(N):
        if t == 0:
            train_model(flat_wr, all_pairs[t], EPOCHS_TEACH, LR_TEACH)
        else:
            replay = sum([all_facts[pt][:REPLAY_N] for pt in range(t)], [])
            train_model(flat_wr, all_pairs[t] + replay, EPOCHS_REPLAY, LR_REPLAY)

    # ── C. Матрёшка ──
    l2 = MLP(D, 64)
    for t in range(N):
        l1 = MLP(D, 16)
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

    # Средняя косинусная близость между ключами внутри каждой темы
    intra_sims = []
    for t in range(N):
        keys = [k for k, v in all_facts[t]]
        sims = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                sims.append(torch.cosine_similarity(keys[i].unsqueeze(0), keys[j].unsqueeze(0)).item())
        intra_sims.append(np.mean(sims) if sims else 0)
    avg_intra = np.mean(intra_sims)

    return (avg_nr, avg_wr, avg_l2), (ok_nr, ok_wr, ok_l2), avg_intra


# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════
print("=" * 72)
print("ЭКСПЕРИМЕНТ 2: Реальные эмбеддинги (nomic-embed-text-v1.5)")
print("=" * 72)

print("\nЗагрузка эмбеддера...")
embedder, emb_dim = get_embedder()

print(f"\nСоздание датасета на {max(TOPIC_COUNTS)} тем...")
all_facts, all_filler = make_topics(embedder, emb_dim, max(TOPIC_COUNTS))

# Вычислим сложность задачи
print("\nСредняя косинусная близость между ключами внутри темы:")
for N in TOPIC_COUNTS:
    sims = []
    for t in range(N):
        keys = [k for k, v in all_facts[t]]
        s = 0
        cnt = 0
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                s += torch.cosine_similarity(keys[i].unsqueeze(0), keys[j].unsqueeze(0)).item()
                cnt += 1
        sims.append(s / cnt if cnt else 0)
    print(f"  {N:3d} тем: средняя внутритемная близость = {np.mean(sims):.4f}")

print(f"\n{'─' * 72}")
print(f"\n{'Темы':>6s}  {'Flat':>8s}  {'Flat+r':>10s}  {'Matryosh':>10s}  "
      f"{'Flat>0.5':>9s}  {'Flat+r>0.5':>10s}  {'Match>0.5':>10s}  {'IntraSim':>9s}")
print("-" * 72)

results = {}
for N in TOPIC_COUNTS:
    avgs, oks, intra = run_experiment(all_facts, all_filler, N)
    results[N] = (avgs, oks, intra)
    print(f"{N:6d}  {avgs[0]:8.4f}  {avgs[1]:10.4f}  {avgs[2]:10.4f}  "
          f"{oks[0]:3d}/{N:<3d}    {oks[1]:3d}/{N:<3d}    {oks[2]:3d}/{N:<3d}    {intra:.4f}")

print(f"\n{'─' * 72}")
print(f"\nСводка (реальные эмбеддинги)\n")
print(f"{'Темы':>6s}  {'Flat':>8s}  {'Flat+r':>10s}  {'Matryoshka':>10s}")
print("-" * 40)
for N in TOPIC_COUNTS:
    avgs, _, _ = results[N]
    print(f"{N:6d}  {avgs[0]:8.4f}  {avgs[1]:10.4f}  {avgs[2]:10.4f}")
