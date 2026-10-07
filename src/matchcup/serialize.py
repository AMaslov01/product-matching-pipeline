from __future__ import annotations

from typing import Any

PRIMARY_FIELDS = ("brand", "model", "article", "type", "size", "quantity", "color")


def _short(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _item_lines(item: dict[str, Any], prefix: str) -> list[str]:
    lines = [
        f"category_{prefix}: {_short(item.get('category'), 100)}",
        f"name_{prefix}: {_short(item.get('name_norm') or item.get('name_raw'), 600)}",
    ]
    for field in PRIMARY_FIELDS:
        value = item.get(f"{field}_norm") or item.get(f"{field}_raw")
        if value:
            lines.append(f"{field}_{prefix}: {_short(value, 220)}")
    remaining = item.get("remaining_attributes")
    if remaining:
        lines.append(f"attributes_{prefix}: {_short(remaining, 1800)}")
    return lines


def serialize_item(item: dict[str, Any]) -> str:
    return "\n".join(_item_lines(item, "item"))


def serialize_pair(item_a: dict[str, Any], item_b: dict[str, Any]) -> tuple[str, str]:
    """Return a symmetric pair representation as tokenizer text/text_pair.

    Items are ordered by stable ID so swapping id1/id2 produces the same transformer input.
    The original id order is preserved only in the output submission.
    """
    if int(item_a["id"]) > int(item_b["id"]):
        item_a, item_b = item_b, item_a
    header = f"category: {_short(item_a.get('category') or item_b.get('category'), 100)}"
    left = "\n".join([header, *_item_lines(item_a, "a")])
    right = "\n".join(_item_lines(item_b, "b"))
    return left, right
