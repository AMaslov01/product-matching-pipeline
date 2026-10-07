from matchcup.normalize import (
    extract_measurements,
    normalize_color,
    normalize_identifier,
    normalize_text,
    transliterate_ru,
)


def test_russian_normalization_and_transliteration() -> None:
    assert normalize_text("  ЧЁРНЫЙ—Samsung  ") == "черный samsung"
    assert transliterate_ru("Самсунг") == "samsung"


def test_identifier_confusables() -> None:
    assert normalize_identifier("АB-С123") == normalize_identifier("AB-C123")


def test_measurement_unit_conversion() -> None:
    assert extract_measurements("бутылка 1 л") == {"volume_ml": [1000.0]}
    assert extract_measurements("1000 ml") == {"volume_ml": [1000.0]}
    assert extract_measurements("2 x 256 GB") == {"storage_mb": [262144.0]}


def test_color_aliases() -> None:
    assert normalize_color("Черный / Black") == "black"
    assert normalize_color("сине-зелёный") == "blue|green"
