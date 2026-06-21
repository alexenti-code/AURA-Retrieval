# AURA Memory Architecture

**Advanced Unified Retrieval Architecture** — архитектура памяти для мультиагентных AI-систем, предотвращающая деградацию знаний через halfLife по scope, композитный скоринг и 6-ролевую когнитивную структуру.

**Дата первой публикации:** Июнь 2026
**Автор:** ООО «АУРУМ ЭСТЕЙТ» / AURA.KIM
**Лицензия:** Apache 2.0

---

## Ключевые компоненты

### 1. Композитный скоринг
Score = α · semantic + β · recency + γ · importance

### 2. HalfLife по scope
- /news, /market → 90 дней
- /objections, /scripts → 730 дней
- /regulations, /rules → 3650 дней

### 3. 6 когнитивных ролей
Colleague (RAG), Intern (эволюционер), Teacher (валидатор), Executor (инженер), Researcher (аудитор), Mentor (guardrail)

### 4. Трёхконтурная модель
Горячий (~200ms) → Тёплый (near-line) → Холодный (offline/nightly)

---

## Публикации
- [Исследование рынка систем памяти](https://aura.kim/research-1.html)
- [Архитектура: 6 ролей и бизнес-кейс](https://aura.kim/research-2.html)  
- [Когнитивные контуры: математика масштабирования](https://aura.kim/research-3.html)
- [Как победить деградацию памяти](https://aura.kim/research-4.html)
- [Преодоление автофагии в мультиагентных LLM-системах](https://aura.kim/research-5.html)
- [Инъекция знаний в LLM: малая модель как бизнес-память](https://aura.kim/research-6.html)
- [LLM + Agent + RAG + AURA: четыре поколения AI-архитектур](https://aura.kim/research-7.html)
- [Инъекция знаний в LLM: обзор рынка 2025–2026](https://aura.kim/research-8.html)
- [Fine-tune в 1000 раз дешевле: двухмодельный конвейер](https://aura.kim/research-9.html)(https://aura.kim/research-1.html)
- [Архитектура: 6 ролей и бизнес-кейс](https://aura.kim/research-2.html)  
- [Когнитивные контуры: математика масштабирования](https://aura.kim/research-3.html)
- [Как победить деградацию памяти](https://aura.kim/research-4.html)

- [Fine-tune в 1000 раз дешевле: двухмодельный конвейер](https://aura.kim/research-9.html)

## Сайт
https://aura.kim/research
