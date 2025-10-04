from app.utils.text_normalizer import TextNormalizer


def test_number_normalization_handles_large_values():
    normalizer = TextNormalizer()
    text = normalizer.normalize("Tôi có 1200 đồng")
    assert "một nghìn hai trăm" in text


def test_currency_shortcuts_are_expanded():
    normalizer = TextNormalizer()
    assert normalizer.normalize("Giá 100k") == "giá một trăm nghìn"


def test_acronym_replacement_from_rules():
    normalizer = TextNormalizer()
    assert normalizer.normalize("Yêu API và NLP") == "yêu ây pi ai và en el pi"
