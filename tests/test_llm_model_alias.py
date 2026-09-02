from alpha_research.llm import effective_model_name


def test_deepseek_official_endpoint_strips_catalog_prefix():
    assert effective_model_name(
        "https://api.deepseek.com/v1", "openai/deepseek-v4-pro"
    ) == "deepseek-v4-pro"


def test_model_alias_is_not_changed_for_other_endpoints():
    assert effective_model_name(
        "https://example.test/v1", "openai/deepseek-v4-pro"
    ) == "openai/deepseek-v4-pro"
