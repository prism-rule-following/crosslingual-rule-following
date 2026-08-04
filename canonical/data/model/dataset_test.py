from config.dataset_config import DatasetConfig, CrossLingualRuleFollowingDataset
import pytest
import urllib

branch = "dataset-design-gen-pipeline"

dataset_loader_load_all = {
    "url": f"https://raw.githubusercontent.com/prism-rule-following/crosslingual-rule-following/{branch}/canonical/data/additional_original_rules_generalised.json",
    "source": "gh",
}

dataset_loader_filter_by_contrastive_pair = {
    "url": f"https://raw.githubusercontent.com/prism-rule-following/crosslingual-rule-following/{branch}/canonical/data/additional_original_rules_generalised.json",
    "source": "gh",
    "contrastive_pair": ["active_cancelled"],
}

dataset_loader_invalid_url = {
    "url": f"https://raw.githubusercontent.com/{branch}/canonical/data/additional_original_rules_generalised.json",
    "source": "gh",
}

dataset_loader_filter_by_grammar_topic = {
    "url": f"https://raw.githubusercontent.com/prism-rule-following/crosslingual-rule-following/{branch}/canonical/data/additional_original_rules_generalised.json",
    "source": "gh",
    "grammar_type": ["imperative", "modal_obligation"],
    "topic": ["mental_health", "mental_health"],
}


def test_custom_dataset_load_from_gh_no_filter():
    config = DatasetConfig.model_validate(dataset_loader_load_all)
    dataset = CrossLingualRuleFollowingDataset(config)
    print(dataset.head(5))
    assert len(dataset) > 0


def test_custom_dataset_filter_by_contrastive_pair():
    config = DatasetConfig.model_validate(dataset_loader_filter_by_contrastive_pair)
    dataset = CrossLingualRuleFollowingDataset(config)
    print(dataset.head(5))
    assert len(dataset) > 0
    assert {"active_cancelled"}.issubset(set(dataset.df["pair_type"].dropna().unique()))


def test_custom_dataset_invalid_url():
    config = DatasetConfig.model_validate(dataset_loader_invalid_url)

    with pytest.raises(urllib.error.URLError):
        CrossLingualRuleFollowingDataset(config)


def test_custom_dataset_filter_by_grammar_topic():
    config = DatasetConfig.model_validate(dataset_loader_filter_by_grammar_topic)
    dataset = CrossLingualRuleFollowingDataset(config)
    print(dataset.head(5))
    assert len(dataset) > 0
    assert {
        "imperative",
        "modal_obligation",
    }.issubset(set(dataset.df["grammar_type"].dropna().unique()))
    assert {"mental_health", "mental_health"}.issubset(
        set(dataset.df["topic"].dropna().unique())
    )
