"""
synthetic.py — генерация синтетических диалогов для тестирования матрёшки.

Сценарий: пользователь и модель обсуждают UI проекта.
В диалог вплетаются факты (ключ→значение), которые потом проверяются на recall.

Формат:
  [
    {"role": "user", "text": "...", "facts": [("key", "val"), ...]},
    {"role": "assistant", "text": "...", "facts": []},
    ...
  ]
"""

import random
from dataclasses import dataclass, field

@dataclass
class Turn:
    role: str          # "user" | "assistant"
    text: str
    facts: list = field(default_factory=list)

@dataclass
class SyntheticDialogue:
    turns: list[Turn]
    topic: str

# Шаблоны реплик для разных тем
TOPIC_TEMPLATES = {
    "ui": {
        "user_utterances": [
            "кнопки должны быть белыми, тёмные отвлекают на светлом фоне",
            "расположи их в левом верхнем углу",
            "шрифт сделай 14pt, не меньше",
            "логотип добавь справа, 120px",
            "я передумал насчёт кнопок — пусть будут синие, как основной бренд",
            "убери тени с карточек, они выглядят грязно",
            "анимацию переключения сделай 300мс ease-out",
        ],
        "assistant_utterances": [
            "хорошо, сделаем кнопки белыми, стандартное решение для research",
            "левый верхний угол, принято",
            "14pt, основной текст, понял",
            "логотип 120px справа",
            "понял, меняем на синие, отменяем белые",
            "тени убраны, карточки плоские",
            "300мс ease-out, добавил",
        ],
        "facts": {
            "color": ("white", "blue"),
            "position": ("top_left", "top_left"),
            "font_size": ("14pt", "14pt"),
            "shadows": ("present", "removed"),
        }
    },
    "prices": {
        "user_utterances": [
            "какая у вас комиссия за продажу квартиры",
            "это дорого, у других агентов 2%",
            "а что входит в эти 3%",
            "хорошо, меня устраивает, когда начинаем",
        ],
        "assistant_utterances": [
            "наша комиссия 3% от суммы сделки",
            "понимаю, но в 3% входит полное сопровождение, фото, реклама",
            "фотосъёмка, реклама на 5 площадках, показы, юрист",
            "отлично, завтра пришлю документы",
        ],
        "facts": {
            "commission": ("asked", "accepted"),
            "includes": ("not_specified", "photography_ads_legal"),
        }
    }
}

def generate_dialogue(topic: str = "ui", num_extra_turns: int = 10) -> SyntheticDialogue:
    """Сгенерировать диалог на тему с перемешанными репликами."""
    template = TOPIC_TEMPLATES[topic]
    turns = []

    # Основные реплики по порядку
    for u, a in zip(template["user_utterances"], template["assistant_utterances"]):
        turns.append(Turn(role="user", text=u))
        turns.append(Turn(role="assistant", text=a))

    # Добавить filler-реплики для растягивания диалога
    fillers = [
        ("давай подождём", "жду"),
        ("есть вопросы?", "нет"),
        ("что думаешь?", "нормально"),
        ("ещё идеи?", "пока нет"),
        ("так, что дальше", "смотри следующий пункт"),
        ("напомни контекст", "мы обсуждаем проект"),
        ("хорошо, продолжим", "продолжаем"),
    ]
    for _ in range(num_extra_turns):
        u, a = random.choice(fillers)
        turns.append(Turn(role="user", text=u))
        turns.append(Turn(role="assistant", text=a))

    # Перемешать в хронологическом порядке (оставить структуру вопрос-ответ)
    # Но не перемешивать пары — только порядок пар
    paired = [(turns[i], turns[i+1]) for i in range(0, len(turns)-1, 2)]
    # Первые 2 пары оставить на месте (начало диалога), остальные перемешать
    if len(paired) > 4:
        head = paired[:2]
        body = paired[2:]
        random.shuffle(body)
        paired = head + body

    result = []
    for u, a in paired:
        result.append(u)
        result.append(a)

    return SyntheticDialogue(turns=result, topic=topic)


def print_dialogue(d: SyntheticDialogue):
    """Вывести диалог в читаемом виде."""
    print(f"\n=== Диалог: {d.topic} ===\n")
    for i, t in enumerate(d.turns):
        prefix = "👤" if t.role == "user" else "🤖"
        tag = f" [{', '.join(t.facts)}]" if t.facts else ""
        print(f"  {prefix} {t.text}{tag}")
    print(f"\n=== Всего реплик: {len(d.turns)} ===")


if __name__ == "__main__":
    d = generate_dialogue("ui", num_extra_turns=20)
    print_dialogue(d)
