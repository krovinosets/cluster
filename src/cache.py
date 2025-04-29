import json
import numpy as np


def save_to_cache(data) -> None:
    """
    Сохраняет данные с преобразованием ndarray в списки
    """

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object {obj} is not JSON serializable")

    with open('cache.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, default=convert, ensure_ascii=False)


def load_from_cache() -> list:
    """
    Загружает данные и преобразует списки обратно в ndarray
    """
    try:
        with open('cache.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            return [np.array(lst) for lst in data]
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def is_cache_not_empty() -> bool:
    """
    Проверяет, содержит ли cache.json какие-либо данные
    """
    try:
        with open('cache.json', 'r', encoding='utf-8') as f:
            return len(json.load(f)) > 0
    except (FileNotFoundError, json.JSONDecodeError):
        return False
