"""Loads the prompt taxonomy YAML and resolves it into runnable prompt configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src import config


@dataclass(frozen=True)
class PromptVariant:
    """A single runnable prompt configuration."""

    id: str  # e.g. "P1b"
    dimension: str  # e.g. "persona"
    label: str
    system_prompt: str
    user_prompt_prefix: str
    user_prompt_suffix: str

    def user_prompt(self, topic: str) -> str:
        """Build the full user prompt by wrapping the topic with prefix/suffix."""
        parts = [p for p in (self.user_prompt_prefix, topic, self.user_prompt_suffix) if p]
        return " ".join(parts)

    @property
    def is_baseline(self) -> bool:
        return self.id.endswith("a")


@dataclass(frozen=True)
class GenParamsVariant:
    """A Tier 6 generation-parameter variant (no prompt text changes)."""

    id: str
    label: str
    temperature: float
    top_p: float

    @property
    def is_baseline(self) -> bool:
        return self.id.endswith("a")


@dataclass(frozen=True)
class CompositePrompt:
    """A composite prompt built from the best variant in each tier."""

    label: str
    system_prompt: str
    user_prompt_prefix: str
    user_prompt_suffix: str
    temperature: float
    top_p: float
    component_ids: list[str]

    def user_prompt(self, topic: str) -> str:
        parts = [p for p in (self.user_prompt_prefix, topic, self.user_prompt_suffix) if p]
        return " ".join(parts)


# ── Dimension names that contain prompt text (not gen params) ──────────────────
PROMPT_DIMENSIONS = ("persona", "structure", "texture", "content", "meta")
GEN_PARAMS_DIMENSION = "gen_params"


class PromptRegistry:
    """Loads taxonomy.yaml and provides lookup methods for the experiment runner."""

    def __init__(self, taxonomy_path: Path | None = None):
        path = taxonomy_path or (config.PROMPTS_DIR / "taxonomy.yaml")
        with open(path) as f:
            self._raw = yaml.safe_load(f)

        self._variants: dict[str, PromptVariant] = {}
        self._gen_params: dict[str, GenParamsVariant] = {}
        self._dimensions: dict[str, list[str]] = {}  # dimension → [variant_ids]
        self._composites: list[CompositePrompt] = []

        self._load_dimensions()
        self._load_composites()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(self, variant_id: str) -> PromptVariant:
        """Look up a single prompt variant by ID (e.g. 'P1b')."""
        return self._variants[variant_id]

    def get_gen_params(self, variant_id: str) -> GenParamsVariant:
        """Look up a single gen-params variant by ID (e.g. 'P6b')."""
        return self._gen_params[variant_id]

    def get_dimension(self, dimension: str) -> list[PromptVariant]:
        """Return all prompt variants in a dimension, including its baseline."""
        return [self._variants[vid] for vid in self._dimensions[dimension]]

    def get_gen_params_variants(self) -> list[GenParamsVariant]:
        """Return all Tier 6 gen-param variants."""
        return list(self._gen_params.values())

    def get_baseline(self, dimension: str) -> PromptVariant:
        """Return the baseline (variant 'a') for a prompt dimension."""
        for vid in self._dimensions[dimension]:
            v = self._variants[vid]
            if v.is_baseline:
                return v
        raise KeyError(f"No baseline found for dimension '{dimension}'")

    def get_gen_params_baseline(self) -> GenParamsVariant:
        """Return the Tier 6 baseline gen-params variant."""
        for gp in self._gen_params.values():
            if gp.is_baseline:
                return gp
        raise KeyError("No baseline found for gen_params dimension")

    @property
    def prompt_dimensions(self) -> list[str]:
        """Names of the prompt-text dimensions (excludes gen_params)."""
        return [d for d in self._dimensions if d != GEN_PARAMS_DIMENSION]

    @property
    def topics(self) -> list[str]:
        return list(self._raw.get("topics", config.ESSAY_TOPICS))

    @property
    def composites(self) -> list[CompositePrompt]:
        return list(self._composites)

    # ── Loading ────────────────────────────────────────────────────────────────

    def _load_dimensions(self) -> None:
        dims = self._raw.get("dimensions", {})
        for dim_name, variants in dims.items():
            ids = []
            for vid, spec in variants.items():
                if dim_name == GEN_PARAMS_DIMENSION:
                    self._gen_params[vid] = GenParamsVariant(
                        id=vid,
                        label=spec["label"],
                        temperature=spec["temperature"],
                        top_p=spec["top_p"],
                    )
                else:
                    self._variants[vid] = PromptVariant(
                        id=vid,
                        dimension=dim_name,
                        label=spec["label"],
                        system_prompt=spec.get("system_prompt", ""),
                        user_prompt_prefix=spec.get("user_prompt_prefix", ""),
                        user_prompt_suffix=spec.get("user_prompt_suffix", ""),
                    )
                ids.append(vid)
            self._dimensions[dim_name] = ids

    def _load_composites(self) -> None:
        for comp in self._raw.get("composite_prompts", []) or []:
            # Merge prompt text from each referenced component
            system_parts = []
            prefix_parts = []
            suffix_parts = []
            for cid in comp["components"]:
                v = self._variants[cid]
                if v.system_prompt:
                    system_parts.append(v.system_prompt)
                if v.user_prompt_prefix:
                    prefix_parts.append(v.user_prompt_prefix)
                if v.user_prompt_suffix:
                    suffix_parts.append(v.user_prompt_suffix)

            gp_id = comp.get("gen_params", "P6a")
            gp = self._gen_params[gp_id]

            self._composites.append(
                CompositePrompt(
                    label=comp["label"],
                    system_prompt=" ".join(system_parts),
                    user_prompt_prefix=" ".join(prefix_parts),
                    user_prompt_suffix=" ".join(suffix_parts),
                    temperature=gp.temperature,
                    top_p=gp.top_p,
                    component_ids=comp["components"],
                )
            )
