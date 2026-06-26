"""
core.py — L0 и L1 быстрые веса для диалоговой памяти.

Уровень 0: одна пара ключ→значение, overwrite при новой паре.
Уровень 1: матрица ранга r, накапливает до N пар как сумму внешних произведений.
           Извлечение: argmin ||W·key - value|| по записанным ассоциациям.

Принцип: W = Σ v_i ⊗ k_i  (сумма внешних произведений, k — ключ, v — значение)
Чтение:  pred = W · k_query
         Если ||pred - любой из записанных value|| < threshold — нашли.
"""

import torch

class FastWeightL0:
    """Быстрый вес уровня 0: одна ассоциация, overwrite при новой записи."""

    def __init__(self, d_model: int):
        self.d_model = d_model
        self.W = torch.zeros(d_model, d_model)
        self._stored_key = None
        self._stored_val = None

    def write(self, key: torch.Tensor, value: torch.Tensor, lr: float = 1.0):
        """Записать пару ключ→значение в матрицу. Один шаг: W ← lr · key ⊗ value."""
        assert key.shape == (self.d_model,) and value.shape == (self.d_model,)
        self.W = lr * torch.outer(value, key)
        self._stored_key = key
        self._stored_val = value

    def read(self, key: torch.Tensor) -> torch.Tensor:
        """Извлечь значение по ключу: v_pred = W · key."""
        return self.W @ key

    def similarity(self, key: torch.Tensor) -> float:
        """Косинусное расстояние между ключом и сохранённым (если есть)."""
        if self._stored_key is None:
            return 0.0
        return torch.cosine_similarity(key.unsqueeze(0), self._stored_key.unsqueeze(0)).item()

    def clear(self):
        self.W.zero_()
        self._stored_key = None
        self._stored_val = None


class FastWeightL1:
    """Быстрый вес уровня 1: накопление ассоциаций через сумму внешних произведений.

    Ёмкость: при ранге r (количество записанных пар) точность retrieval падает
    после N ≈ 2r из-за superposition (Elhage et al., 2022).
    """

    def __init__(self, d_model: int, retention_threshold: float = 0.8):
        self.d_model = d_model
        self.W = torch.zeros(d_model, d_model)
        self.keys = []        # для оценки точности, не для чтения
        self.values = []      # для оценки точности, не для чтения
        self.threshold = retention_threshold

    def write(self, key: torch.Tensor, value: torch.Tensor):
        """Добавить пару: W += v ⊗ k (так что W @ k ≈ v при retrieval)."""
        assert key.shape == (self.d_model,) and value.shape == (self.d_model,)
        self.W += torch.outer(value, key)
        self.keys.append(key)
        self.values.append(value)

    def read(self, key: torch.Tensor) -> torch.Tensor:
        """Извлечь: v_pred = W · key."""
        return self.W @ key

    def recall(self, key_idx: int) -> float:
        """Recall для одной ассоциации: косинус между извлечённым и оригиналом."""
        k = self.keys[key_idx]
        v_orig = self.values[key_idx]
        v_pred = self.read(k)
        return torch.cosine_similarity(v_pred.unsqueeze(0), v_orig.unsqueeze(0)).item()

    def average_recall(self) -> float:
        """Средний recall по всем записанным ассоциациям."""
        if not self.keys:
            return 0.0
        return sum(self.recall(i) for i in range(len(self.keys))) / len(self.keys)

    def count_above_threshold(self) -> int:
        """Сколько ассоциаций имеют recall >= threshold."""
        if not self.keys:
            return 0
        return sum(1 for i in range(len(self.keys)) if self.recall(i) >= self.threshold)

    def estimate_capacity(self, max_n: int = 2000) -> int:
        """Оценить реальную ёмкость: сколько случайных пар держит recall >= threshold."""
        original_W = self.W.clone()
        self.clear()
        good = 0
        for i in range(max_n):
            k = torch.randn(self.d_model)
            v = torch.randn(self.d_model)
            k = k / k.norm()
            v = v / v.norm()
            self.write(k, v)
            sim = self.recall(i)
            if sim >= self.threshold:
                good += 1
            else:
                break
        self.W = original_W
        return good

    def clear(self):
        self.W.zero_()
        self.keys.clear()
        self.values.clear()


def random_pair(d_model: int):
    """Сгенерировать случайную пару ключ→значение (нормализованные векторы)."""
    k = torch.randn(d_model)
    v = torch.randn(d_model)
    return k / k.norm(), v / v.norm()


class FastWeightL2:
    """Уровень 2 (сессия): принимает weights от L1 при смене темы.
    В отличие от L1, который накапливает пары, L2 накапливает целые
    тематические блоки — для этого он хранит не отдельные пары, а
    объединяет W от нескольких L1 через взвешенное сложение.
    """

    def __init__(self, d_model: int):
        self.d_model = d_model
        self.W = torch.zeros(d_model, d_model)
        self.sources = []  # метки тем для диагностики

    def absorb(self, l1: FastWeightL1, topic_label: str):
        """Впитать W от L1 при смене темы."""
        self.W += l1.W
        self.sources.append(topic_label)
        l1.clear()

    def read(self, key: torch.Tensor) -> torch.Tensor:
        return self.W @ key
