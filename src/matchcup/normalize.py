from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[0-9a-zа-я]+", re.IGNORECASE)
_IDENTIFIER_RE = re.compile(r"(?=[0-9a-zа-я-]*\d)[0-9a-zа-я]+(?:[-_/][0-9a-zа-я]+)*", re.I)
_NUMBER_UNIT_RE = re.compile(
    r"(?<![\w])(?P<number>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>мг|mg|кг|kg|гр|г|g|мл|ml|л|l|мм|mm|см|cm|м|m|"
    r"кб|kb|мб|mb|гб|gb|тб|tb|шт|pcs?|уп|pack|пар(?:а|ы)?|компл(?:ект|екта)?)\b",
    re.I,
)

_RU_TRANSLIT = str.maketrans(
    {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "е": "e",
        "ё": "e",
        "ж": "zh",
        "з": "z",
        "и": "i",
        "й": "i",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "kh",
        "ц": "ts",
        "ч": "ch",
        "ш": "sh",
        "щ": "shch",
        "ъ": "",
        "ы": "y",
        "ь": "",
        "э": "e",
        "ю": "yu",
        "я": "ya",
    }
)

_CONFUSABLE_TO_LATIN = str.maketrans(
    {
        "а": "a",
        "в": "b",
        "с": "c",
        "е": "e",
        "н": "h",
        "к": "k",
        "м": "m",
        "о": "o",
        "р": "p",
        "т": "t",
        "х": "x",
        "у": "y",
    }
)

_UNITS: dict[str, tuple[str, float]] = {
    "мг": ("mass_g", 0.001),
    "mg": ("mass_g", 0.001),
    "г": ("mass_g", 1.0),
    "гр": ("mass_g", 1.0),
    "g": ("mass_g", 1.0),
    "кг": ("mass_g", 1000.0),
    "kg": ("mass_g", 1000.0),
    "мл": ("volume_ml", 1.0),
    "ml": ("volume_ml", 1.0),
    "л": ("volume_ml", 1000.0),
    "l": ("volume_ml", 1000.0),
    "мм": ("length_mm", 1.0),
    "mm": ("length_mm", 1.0),
    "см": ("length_mm", 10.0),
    "cm": ("length_mm", 10.0),
    "м": ("length_mm", 1000.0),
    "m": ("length_mm", 1000.0),
    "кб": ("storage_mb", 1 / 1024),
    "kb": ("storage_mb", 1 / 1024),
    "мб": ("storage_mb", 1.0),
    "mb": ("storage_mb", 1.0),
    "гб": ("storage_mb", 1024.0),
    "gb": ("storage_mb", 1024.0),
    "тб": ("storage_mb", 1024.0 * 1024.0),
    "tb": ("storage_mb", 1024.0 * 1024.0),
    "шт": ("count", 1.0),
    "pc": ("count", 1.0),
    "pcs": ("count", 1.0),
    "уп": ("pack_count", 1.0),
    "pack": ("pack_count", 1.0),
    "пар": ("pair_count", 1.0),
    "пара": ("pair_count", 1.0),
    "пары": ("pair_count", 1.0),
    "компл": ("set_count", 1.0),
    "комплект": ("set_count", 1.0),
    "комплекта": ("set_count", 1.0),
}

_COLOR_ALIASES = {
    "black": "black",
    "черный": "black",
    "white": "white",
    "белый": "white",
    "red": "red",
    "красный": "red",
    "blue": "blue",
    "синий": "blue",
    "голубой": "blue",
    "green": "green",
    "зеленый": "green",
    "grey": "gray",
    "gray": "gray",
    "серый": "gray",
    "silver": "silver",
    "серебристый": "silver",
    "gold": "gold",
    "золотой": "gold",
    "beige": "beige",
    "бежевый": "beige",
    "brown": "brown",
    "коричневый": "brown",
    "pink": "pink",
    "розовый": "pink",
    "purple": "purple",
    "фиолетовый": "purple",
    "orange": "orange",
    "оранжевый": "orange",
    "yellow": "yellow",
    "желтый": "yellow",
    "transparent": "transparent",
    "прозрачный": "transparent",
    "multicolor": "multicolor",
    "разноцветный": "multicolor",
}

_COLOR_STEMS = {
    "черн": "black",
    "бел": "white",
    "красн": "red",
    "син": "blue",
    "голуб": "blue",
    "зелен": "green",
    "сер": "gray",
    "серебр": "silver",
    "золот": "gold",
    "беж": "beige",
    "коричн": "brown",
    "розов": "pink",
    "фиолет": "purple",
    "оранж": "orange",
    "желт": "yellow",
    "прозрачн": "transparent",
    "разноцвет": "multicolor",
}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = unicodedata.normalize("NFKC", str(value)).casefold().replace("ё", "е")
    text = "".join(ch if ch.isalnum() else " " for ch in text)
    return _SPACE_RE.sub(" ", text).strip()


def transliterate_ru(value: str) -> str:
    return normalize_text(value).translate(_RU_TRANSLIT)


def normalize_identifier(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().replace("ё", "е")
    text = text.translate(_CONFUSABLE_TO_LATIN)
    return "".join(ch for ch in text if ch.isalnum())


def tokens(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(normalize_text(value)))


def identifier_tokens(value: str) -> list[str]:
    result = {normalize_identifier(token) for token in _IDENTIFIER_RE.findall(value or "")}
    return sorted(
        token for token in result if len(token) >= 2 and any(ch.isdigit() for ch in token)
    )


def numeric_tokens(value: str) -> list[str]:
    return sorted(
        {
            token
            for token in _TOKEN_RE.findall(normalize_text(value))
            if any(c.isdigit() for c in token)
        }
    )


def _clean_float(value: float) -> float:
    if math.isclose(value, round(value), rel_tol=0, abs_tol=1e-9):
        return float(round(value))
    return round(value, 6)


def extract_measurements(*values: Any) -> dict[str, list[float]]:
    result: dict[str, set[float]] = {}
    for raw in values:
        text = unicodedata.normalize("NFKC", str(raw or "")).casefold().replace("ё", "е")
        for match in _NUMBER_UNIT_RE.finditer(text):
            number = float(match.group("number").replace(",", "."))
            unit = match.group("unit").lower()
            kind, factor = _UNITS[unit]
            result.setdefault(kind, set()).add(_clean_float(number * factor))
    return {key: sorted(values) for key, values in sorted(result.items())}


def normalize_size(value: Any, key: str = "") -> str:
    norm = normalize_text(value)
    if not norm:
        return ""
    key_norm = normalize_text(key)
    if "российск" in key_norm or re.search(r"\bru\b", key_norm):
        system = "ru"
    elif "европ" in key_norm or re.search(r"\beu\b", key_norm):
        system = "eu"
    elif "америк" in key_norm or re.search(r"\bus\b", key_norm):
        system = "us"
    elif "международ" in key_norm:
        system = "intl"
    else:
        system = "unknown"
    values = sorted(set(_TOKEN_RE.findall(norm)))
    return f"{system}:" + "|".join(values)


def normalize_color(value: Any) -> str:
    norm = normalize_text(value)
    aliases: set[str] = set()
    for token in _TOKEN_RE.findall(norm):
        if token in _COLOR_ALIASES:
            aliases.add(_COLOR_ALIASES[token])
        aliases.update(color for stem, color in _COLOR_STEMS.items() if token.startswith(stem))
    return "|".join(sorted(aliases)) if aliases else norm


@dataclass(frozen=True)
class FieldCandidate:
    score: int
    key: str
    raw: str


def _key_score(field: str, key: str) -> int:
    key = normalize_text(key)
    if field == "type":
        return 100 if key == "тип" else 0
    if field == "brand":
        if key == "бренд":
            return 100
        if key == "бренд в одежде и обуви":
            return 95
        if key == "марка":
            return 80
        if key in {"производитель", "изготовитель"}:
            return 45
        return 60 if "бренд" in key else 0
    if field == "model":
        if key == "модель":
            return 100
        if key in {"название модели", "модель товара"}:
            return 95
        if "модель" in key and not any(x in key for x in ("совместим", "устройств", "автомоб")):
            return 60
        return 0
    if field == "article":
        if key == "артикул":
            return 100
        if any(x in key for x in ("партномер", "part number", "oem")):
            return 95
        return 70 if "артикул" in key else 0
    if field == "color":
        if key == "цвет товара":
            return 100
        if key == "название цвета":
            return 95
        if key == "цвет":
            return 90
        if "цвет" in key and not any(x in key for x in ("температур", "свечен", "подсвет")):
            return 50
        return 0
    if field == "size":
        if any(x in key for x in ("упаков", "габарит", "размеры мм", "размер файла")):
            return 0
        if key == "российский размер":
            return 100
        if key in {"размер производителя", "размер обуви", "размер одежды"}:
            return 95
        if key == "размер":
            return 90
        return 50 if "размер" in key else 0
    if field == "quantity":
        exact = {
            "единиц в одном товаре",
            "количество в упаковке шт",
            "количество предметов в комплекте",
            "количество штук",
            "количество шт",
            "число штук",
            "количество упаковок",
        }
        if key in exact:
            return 100
        if "количество" in key and not any(x in key for x in ("измерен", "режим", "скорост")):
            return 50
        return 0
    return 0


def select_fields(attributes: Mapping[str, Any]) -> dict[str, FieldCandidate | None]:
    selected: dict[str, FieldCandidate | None] = {
        field: None for field in ("type", "brand", "model", "article", "color", "size", "quantity")
    }
    for raw_key, value in attributes.items():
        raw = str(value) if value is not None else ""
        if not raw.strip():
            continue
        for field in selected:
            score = _key_score(field, raw_key)
            current = selected[field]
            if score and (current is None or score > current.score):
                selected[field] = FieldCandidate(score, str(raw_key), raw)
    return selected


def safe_attributes(value: Any) -> tuple[dict[str, Any], int]:
    if isinstance(value, Mapping):
        return dict(value), 0
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}, 1
    return (parsed, 0) if isinstance(parsed, dict) else ({}, 1)


def compact_remaining_attributes(
    attributes: Mapping[str, Any], selected_keys: Iterable[str], max_chars: int
) -> str:
    excluded = {normalize_text(key) for key in selected_keys}
    parts: list[str] = []
    length = 0
    for raw_key, raw_value in sorted(attributes.items(), key=lambda pair: normalize_text(pair[0])):
        key = normalize_text(raw_key)
        value = normalize_text(raw_value)
        if not key or not value or key in excluded:
            continue
        part = f"{key}={value}"
        if length + len(part) + 2 > max_chars:
            break
        parts.append(part)
        length += len(part) + 2
    return "; ".join(parts)
