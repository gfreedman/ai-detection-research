"""
Tests for PromptRegistry -- loads the real ``taxonomy.yaml``.

Covers:

- Clean loading of the taxonomy file.
- Dimension enumeration (Tiers 1-5 only, excludes gen_params).
- Topic loading (five control topics).
- Variant lookup by ID, dimension, and baseline.
- Gen-params variant loading and baseline lookup.
- User prompt assembly (prefix + topic + suffix).
- Composite prompt loading and merge logic.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from src.prompt_registry import (
    CompositePrompt,
    GenParamsVariant,
    PromptRegistry,
    PromptVariant,
)

# Path to the real taxonomy file used by most tests.
TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "prompts" / "taxonomy.yaml"


@pytest.fixture
def registry() -> PromptRegistry:
    """Load the real taxonomy.yaml for test use."""
    return PromptRegistry(TAXONOMY_PATH)


# ===========================================================================
# Basic loading
# ===========================================================================


def test_loads_without_error(registry):
    """The real ``taxonomy.yaml`` should load cleanly without raising."""

    assert registry is not None


def test_prompt_dimensions_excludes_gen_params(registry):
    """``prompt_dimensions`` should list the 5 text dimensions, NOT gen_params."""

    dims = registry.prompt_dimensions

    assert "gen_params" not in dims
    assert set(dims) == {"persona", "structure", "texture", "content", "meta"}


def test_topics_loaded(registry):
    """Should load all 5 essay topics from the taxonomy."""

    assert len(registry.topics) == 5
    assert "social media" in registry.topics[0].lower()


# ===========================================================================
# Variant lookup
# ===========================================================================


def test_get_variant_by_id(registry):
    """Looking up P1b should return the 'Teen student' persona variant."""

    v = registry.get("P1b")

    assert isinstance(v, PromptVariant)
    assert v.id == "P1b"
    assert v.dimension == "persona"
    assert v.label == "Teen student"
    assert "Alex" in v.system_prompt


def test_get_baseline(registry):
    """The persona baseline (P1a) should have no system prompt."""

    b = registry.get_baseline("persona")

    assert b.id == "P1a"
    assert b.is_baseline
    assert b.system_prompt == ""


def test_get_dimension_returns_all_variants(registry):
    """``get_dimension("texture")`` should return all 5 texture variants in order."""

    variants = registry.get_dimension("texture")
    ids = [v.id for v in variants]

    assert ids == ["P3a", "P3b", "P3c", "P3d", "P3e"]
    assert all(isinstance(v, PromptVariant) for v in variants)


def test_every_dimension_has_baseline(registry):
    """Every prompt dimension must have a baseline variant with empty prompts."""

    for dim in registry.prompt_dimensions:
        b = registry.get_baseline(dim)

        assert b.is_baseline, f"Baseline for '{dim}' does not end with 'a'"
        assert b.system_prompt == ""
        assert b.user_prompt_suffix == ""


def test_unknown_variant_raises(registry):
    """Looking up a non-existent variant ID should raise KeyError."""

    with pytest.raises(KeyError):
        registry.get("P99z")


# ===========================================================================
# Gen-params variants (Tier 6)
# ===========================================================================


def test_gen_params_loaded(registry):
    """Should load all 4 gen-params variants (P6a-P6d)."""

    variants = registry.get_gen_params_variants()

    assert len(variants) == 4
    assert all(isinstance(v, GenParamsVariant) for v in variants)


def test_gen_params_baseline(registry):
    """The gen-params baseline (P6a) should have temperature=0.7, top_p=0.9."""

    b = registry.get_gen_params_baseline()

    assert b.id == "P6a"
    assert b.temperature == 0.7
    assert b.top_p == 0.9


def test_gen_params_lookup(registry):
    """Looking up P6c should return temperature=1.3, top_p=0.98."""

    v = registry.get_gen_params("P6c")

    assert v.temperature == 1.3
    assert v.top_p == 0.98


# ===========================================================================
# User prompt assembly
# ===========================================================================


def test_baseline_user_prompt_is_just_topic(registry):
    """Baseline variants (no prefix/suffix) should return the topic verbatim."""

    b = registry.get_baseline("persona")
    topic = "Write a 500-word essay about testing."

    assert b.user_prompt(topic) == topic


def test_suffix_appended_to_topic(registry):
    """P2b (Avoid 5-paragraph) should append its suffix after the topic."""

    v = registry.get("P2b")
    topic = "Write a 500-word essay about testing."
    prompt = v.user_prompt(topic)

    assert prompt.startswith(topic)
    assert "5-paragraph" in prompt


def test_empty_prefix_and_suffix_omitted(registry):
    """When prefix and suffix are both empty, user_prompt should not add whitespace."""

    b = registry.get_baseline("structure")
    topic = "Write an essay."

    assert b.user_prompt(topic) == "Write an essay."


# ===========================================================================
# Composite prompts
# ===========================================================================


def test_composites_loaded(registry):
    """Composite prompts defined in taxonomy.yaml are loaded correctly."""

    assert len(registry.composites) == 3
    labels = [c.label for c in registry.composites]
    assert "Top 3 (persona + structure + texture)" in labels
    assert "All 5 winners" in labels


def test_composite_loading(tmp_path):
    """
    Composites should merge system_prompt and suffixes from their component
    variants and adopt temperature/top_p from the specified gen-params variant.
    """

    # -- Build a minimal taxonomy with two dimensions + one composite -------
    taxonomy = {
        "dimensions": {
            "persona": {
                "P1a": {"label": "None", "system_prompt": "", "user_prompt_prefix": "", "user_prompt_suffix": ""},
                "P1b": {"label": "Teen", "system_prompt": "You are Alex.", "user_prompt_prefix": "", "user_prompt_suffix": ""},
            },
            "texture": {
                "P3a": {"label": "None", "system_prompt": "", "user_prompt_prefix": "", "user_prompt_suffix": ""},
                "P3c": {"label": "Hedging", "system_prompt": "", "user_prompt_prefix": "", "user_prompt_suffix": "Use hedging language."},
            },
            "gen_params": {
                "P6a": {"label": "Baseline", "temperature": 0.7, "top_p": 0.9},
                "P6b": {"label": "Moderate", "temperature": 1.0, "top_p": 0.95},
            },
        },
        "topics": ["Write an essay."],
        "composite_prompts": [
            {
                "label": "Teen + Hedging @ moderate",
                "components": ["P1b", "P3c"],
                "gen_params": "P6b",
            }
        ],
    }

    path = tmp_path / "taxonomy.yaml"
    path.write_text(yaml.dump(taxonomy))

    reg = PromptRegistry(path)
    comps = reg.composites

    assert len(comps) == 1

    c = comps[0]
    assert isinstance(c, CompositePrompt)
    assert c.label == "Teen + Hedging @ moderate"
    assert "Alex" in c.system_prompt
    assert c.temperature == 1.0
    assert c.top_p == 0.95

    # The assembled prompt should contain both the topic and the hedging suffix.
    prompt = c.user_prompt("Write an essay.")

    assert "Write an essay." in prompt
    assert "hedging" in prompt.lower()
