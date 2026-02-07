"""
Loads the prompt taxonomy YAML and resolves it into runnable prompt configs.

The taxonomy file (``prompts/taxonomy.yaml``) defines every prompt variant as
structured data rather than hard-coded strings scattered through Python code.
This module parses that YAML into typed data classes that the experiment runner
iterates over.

Data Classes
------------
- :class:`PromptVariant`   -- a single prompt-text variant (Tiers 1-5).
- :class:`GenParamsVariant` -- a Tier 6 generation-parameter variant.
- :class:`CompositePrompt` -- a post-ablation combo of the best per-tier variants.

Registry
--------
- :class:`PromptRegistry`  -- loads ``taxonomy.yaml`` and exposes lookup helpers
  for dimensions, variants, baselines, topics, and composites.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src import config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Dimension names that contain prompt text (Tiers 1-5).
PROMPT_DIMENSIONS = ("persona", "structure", "texture", "content", "meta")

# The special dimension whose variants modify generation parameters, not prompt text.
GEN_PARAMS_DIMENSION = "gen_params"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptVariant:
    """
    A single runnable prompt configuration from one of the text dimensions
    (Tiers 1-5).

    The full user prompt is assembled at runtime by wrapping the topic string
    with an optional prefix and suffix.

    @param id:                 Short identifier, e.g. ``"P1b"``.
    @param dimension:          Parent dimension name, e.g. ``"persona"``.
    @param label:              Human-readable label for logging and plots.
    @param system_prompt:      Text sent as the Gemini system instruction
                               (empty string = no system instruction).
    @param user_prompt_prefix: Text prepended before the topic.
    @param user_prompt_suffix: Text appended after the topic.
    """

    id: str
    dimension: str
    label: str
    system_prompt: str
    user_prompt_prefix: str
    user_prompt_suffix: str

    def user_prompt(self, topic: str) -> str:
        """
        Build the full user prompt by wrapping *topic* with prefix/suffix.

        Empty prefix or suffix values are silently omitted so the resulting
        string never has leading/trailing whitespace artefacts.

        @param topic: The essay topic string (one of the five control topics).
        @returns:     The assembled user prompt.
        """

        parts = [
            p for p in (self.user_prompt_prefix, topic, self.user_prompt_suffix)
            if p
        ]

        return " ".join(parts)

    @property
    def is_baseline(self) -> bool:
        """
        Whether this variant is the baseline for its dimension.

        Convention: baseline variant IDs always end with ``'a'``
        (e.g. ``P1a``, ``P2a``).
        """
        return self.id.endswith("a")


@dataclass(frozen=True)
class GenParamsVariant:
    """
    A Tier 6 generation-parameter variant (no prompt text changes).

    These vary ``temperature`` and ``top_p`` while keeping the prompt text
    fixed at the baseline.

    @param id:          Short identifier, e.g. ``"P6b"``.
    @param label:       Human-readable label for logging and plots.
    @param temperature: Sampling temperature for this variant.
    @param top_p:       Nucleus-sampling threshold for this variant.
    """

    id: str
    label: str
    temperature: float
    top_p: float

    @property
    def is_baseline(self) -> bool:
        """Whether this is the Tier 6 baseline (``P6a``)."""
        return self.id.endswith("a")


@dataclass(frozen=True)
class CompositePrompt:
    """
    A composite prompt built by combining the best variant from each tier after
    the ablation study.

    Composites merge the ``system_prompt``, ``user_prompt_prefix``, and
    ``user_prompt_suffix`` from each referenced component, and adopt the
    generation parameters of a specified Tier 6 variant.

    @param label:              Human-readable label.
    @param system_prompt:      Merged system instruction from all components.
    @param user_prompt_prefix: Merged prefix from all components.
    @param user_prompt_suffix: Merged suffix from all components.
    @param temperature:        Temperature from the chosen gen-params variant.
    @param top_p:              top_p from the chosen gen-params variant.
    @param component_ids:      List of variant IDs that were combined.
    """

    label: str
    system_prompt: str
    user_prompt_prefix: str
    user_prompt_suffix: str
    temperature: float
    top_p: float
    component_ids: list[str]

    def user_prompt(self, topic: str) -> str:
        """
        Build the full user prompt by wrapping *topic* with prefix/suffix.

        @param topic: The essay topic string.
        @returns:     The assembled user prompt.
        """

        parts = [
            p for p in (self.user_prompt_prefix, topic, self.user_prompt_suffix)
            if p
        ]

        return " ".join(parts)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class PromptRegistry:
    """
    Loads ``taxonomy.yaml`` and provides lookup methods for the experiment runner.

    On construction, the registry parses the YAML file and builds three internal
    indexes:

    - ``_variants``   -- all Tier 1-5 prompt variants, keyed by ID.
    - ``_gen_params`` -- all Tier 6 gen-param variants, keyed by ID.
    - ``_composites`` -- composite prompts (populated after ablation results).
    """

    def __init__(self, taxonomy_path: Path | None = None):
        """
        Load and parse the taxonomy YAML.

        @param taxonomy_path: Path to ``taxonomy.yaml``.  Defaults to
                              ``config.PROMPTS_DIR / "taxonomy.yaml"``.
        """

        path = taxonomy_path or (config.PROMPTS_DIR / "taxonomy.yaml")

        with open(path) as f:
            self._raw = yaml.safe_load(f)

        # -- Internal indexes -----------------------------------------------
        self._variants: dict[str, PromptVariant] = {}
        self._gen_params: dict[str, GenParamsVariant] = {}
        self._dimensions: dict[str, list[str]] = {}  # dimension -> [variant_ids]
        self._composites: list[CompositePrompt] = []

        # -- Populate indexes from the parsed YAML --------------------------
        self._load_dimensions()
        self._load_composites()

    # ── Public API ─────────────────────────────────────────────────────────

    def get(self, variant_id: str) -> PromptVariant:
        """
        Look up a single prompt variant by ID.

        @param variant_id: E.g. ``"P1b"``, ``"P3c"``.
        @returns:          The matching :class:`PromptVariant`.
        @raises KeyError:  If the ID does not exist.
        """
        return self._variants[variant_id]

    def get_gen_params(self, variant_id: str) -> GenParamsVariant:
        """
        Look up a single gen-params variant by ID.

        @param variant_id: E.g. ``"P6b"``.
        @returns:          The matching :class:`GenParamsVariant`.
        @raises KeyError:  If the ID does not exist.
        """
        return self._gen_params[variant_id]

    def get_dimension(self, dimension: str) -> list[PromptVariant]:
        """
        Return all prompt variants in a dimension, including its baseline.

        @param dimension: E.g. ``"persona"``, ``"texture"``.
        @returns:         A list of :class:`PromptVariant` objects.
        @raises KeyError: If the dimension does not exist.
        """
        return [self._variants[vid] for vid in self._dimensions[dimension]]

    def get_gen_params_variants(self) -> list[GenParamsVariant]:
        """
        Return all Tier 6 gen-param variants.

        @returns: A list of :class:`GenParamsVariant` objects.
        """
        return list(self._gen_params.values())

    def get_baseline(self, dimension: str) -> PromptVariant:
        """
        Return the baseline (variant ``'a'``) for a prompt dimension.

        @param dimension: E.g. ``"persona"``.
        @returns:         The baseline :class:`PromptVariant`.
        @raises KeyError: If the dimension has no baseline variant.
        """

        for vid in self._dimensions[dimension]:
            v = self._variants[vid]
            if v.is_baseline:
                return v

        raise KeyError(f"No baseline found for dimension '{dimension}'")

    def get_gen_params_baseline(self) -> GenParamsVariant:
        """
        Return the Tier 6 baseline gen-params variant (``P6a``).

        @returns:         The baseline :class:`GenParamsVariant`.
        @raises KeyError: If no baseline exists in the gen_params dimension.
        """

        for gp in self._gen_params.values():
            if gp.is_baseline:
                return gp

        raise KeyError("No baseline found for gen_params dimension")

    @property
    def prompt_dimensions(self) -> list[str]:
        """
        Names of the prompt-text dimensions (Tiers 1-5).

        Excludes the ``gen_params`` dimension which is handled separately.
        """
        return [d for d in self._dimensions if d != GEN_PARAMS_DIMENSION]

    @property
    def topics(self) -> list[str]:
        """
        The list of essay topics defined in the taxonomy.

        Falls back to ``config.ESSAY_TOPICS`` if the YAML has none.
        """
        return list(self._raw.get("topics", config.ESSAY_TOPICS))

    @property
    def composites(self) -> list[CompositePrompt]:
        """
        Composite prompts defined in the taxonomy (empty until ablation results
        are used to fill them in).
        """
        return list(self._composites)

    # ── Loading internals ──────────────────────────────────────────────────

    def _load_dimensions(self) -> None:
        """
        Parse the ``dimensions`` block from the YAML and populate the variant
        and gen-params indexes.

        Each dimension maps to a dict of variant specs.  Tier 6 variants
        (``gen_params``) are stored separately from Tier 1-5 prompt variants.
        """

        dims = self._raw.get("dimensions", {})

        for dim_name, variants in dims.items():
            ids = []

            for vid, spec in variants.items():

                if dim_name == GEN_PARAMS_DIMENSION:
                    # Tier 6: generation parameters only, no prompt text.
                    self._gen_params[vid] = GenParamsVariant(
                        id=vid,
                        label=spec["label"],
                        temperature=spec["temperature"],
                        top_p=spec["top_p"],
                    )
                else:
                    # Tiers 1-5: prompt text variants.
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
        """
        Parse the ``composite_prompts`` block from the YAML.

        Each composite merges the system prompt, prefix, and suffix from its
        referenced component variants and adopts temperature/top_p from a
        specified gen-params variant (defaulting to ``P6a``).
        """

        for comp in self._raw.get("composite_prompts", []) or []:

            # -- Merge prompt text from each component variant ---------------
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

            # -- Look up the gen-params variant (default to baseline) --------
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
