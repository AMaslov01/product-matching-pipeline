import json

from matchcup.features import pair_features
from matchcup.parser import canonicalize_record
from matchcup.serialize import serialize_pair


def item(item_id: int, name: str, attributes: dict, category: str = "Электроника") -> dict:
    return canonicalize_record(
        {
            "id": item_id,
            "name": name,
            "attributes": json.dumps(attributes, ensure_ascii=False),
            "category": category,
        }
    )


def test_parser_selects_and_preserves_typed_fields() -> None:
    parsed = item(
        1,
        "Samsung S24 256 GB Black",
        {"Бренд": "Samsung", "Артикул": "SM-S921", "Цвет товара": "Черный", "Память": "256 ГБ"},
    )
    assert parsed["brand_norm"] == "samsung"
    assert parsed["article_norm"] == "sms921"
    assert parsed["color_norm"] == "black"
    assert json.loads(parsed["measurements"])["storage_mb"] == [262144.0]
    assert parsed["parse_error"] == 0


def test_invalid_json_is_nonfatal() -> None:
    parsed = canonicalize_record(
        {"id": 1, "name": "test", "attributes": "{broken", "category": "Дом и сад"}
    )
    assert parsed["parse_error"] == 1
    assert parsed["attribute_count"] == 0


def test_serialization_is_pair_symmetric() -> None:
    left = item(2, "Самсунг S24", {"Бренд": "Самсунг"})
    right = item(1, "Samsung S24", {"Бренд": "Samsung"})
    assert serialize_pair(left, right) == serialize_pair(right, left)


def test_features_detect_brand_alias_and_storage_conflict() -> None:
    left = item(1, "Самсунг S24 256 ГБ", {"Бренд": "Самсунг", "Артикул": "S24"})
    right = item(2, "Samsung S24 512 GB", {"Бренд": "Samsung", "Артикул": "S24"})
    features = pair_features(left, right)
    assert features["brand_translit_equal"] == 1.0
    assert features["article_equal"] == 1.0
    assert features["measurement_conflict_types"] >= 1.0


def test_pair_features_are_symmetric() -> None:
    left = item(1, "Самсунг S24 256 ГБ", {"Бренд": "Самсунг", "Артикул": "S24"})
    right = item(
        2,
        "Samsung S24 512 GB",
        {"Бренд": "Samsung", "Артикул": "S24", "Цвет товара": "Черный"},
    )
    assert pair_features(left, right) == pair_features(right, left)


def test_features_align_remaining_attributes_by_key() -> None:
    left = item(
        1,
        "Телефон 256 ГБ",
        {"Бренд": "Test", "NFC": "Да", "Материал": "Сталь", "Память": "256 ГБ"},
    )
    right = item(
        2,
        "Телефон 256 GB",
        {"Бренд": "Test", "NFC": "Да", "Материал": "Пластик", "Память": "256 GB"},
    )
    features = pair_features(left, right)
    assert features["aligned_attribute_key_common"] == 3.0
    assert features["aligned_attribute_value_exact"] == 1.0
    assert features["aligned_attribute_value_conflict"] == 2.0
    assert features["aligned_attribute_numeric_overlap"] == 1.0
