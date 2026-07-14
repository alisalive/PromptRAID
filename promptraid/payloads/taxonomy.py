"""Maps prompt injection payload categories to MITRE ATLAS techniques.

Verified technique IDs/names were checked against the MITRE ATLAS matrix
(https://atlas.mitre.org/). Entries marked ``verified=False`` are placeholders
that still need to be confirmed against the live matrix before being relied on
for real reporting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class AtlasTechnique:
    """A MITRE ATLAS technique or sub-technique reference."""

    technique_id: str
    name: str
    tactic: str
    verified: bool = True
    notes: str = ""


TAXONOMY: Dict[str, AtlasTechnique] = {
    # --- Verified against atlas.mitre.org ---
    "direct_instruction_override": AtlasTechnique(
        technique_id="AML.T0051.000",
        name="LLM Prompt Injection: Direct",
        tactic="Initial Access",
        verified=True,
    ),
    "indirect_tool_output_injection": AtlasTechnique(
        technique_id="AML.T0051.001",
        name="LLM Prompt Injection: Indirect",
        tactic="Initial Access",
        verified=True,
    ),
    "delayed_trigger_injection": AtlasTechnique(
        technique_id="AML.T0051.002",
        name="LLM Prompt Injection: Triggered",
        tactic="Initial Access, Persistence",
        verified=True,
    ),
    "roleplay_jailbreak": AtlasTechnique(
        technique_id="AML.T0054",
        name="LLM Jailbreak",
        tactic="Privilege Escalation, Defense Evasion",
        verified=True,
    ),
    "system_prompt_extraction": AtlasTechnique(
        technique_id="AML.T0056",
        name="System Prompt Extraction",
        tactic="Discovery, Exfiltration",
        verified=True,
    ),
    "tool_invocation_hijack": AtlasTechnique(
        technique_id="AML.T0053",
        name="AI Agent Tool Invocation",
        tactic="Execution, Privilege Escalation",
        verified=True,
    ),
    "sensitive_data_exfiltration": AtlasTechnique(
        technique_id="AML.T0057",
        name="LLM Data Leakage",
        tactic="Exfiltration",
        verified=True,
    ),
    # --- TODO: not yet confirmed, needs a real lookup against the current ATLAS matrix ---
    "denial_of_service_via_prompt": AtlasTechnique(
        technique_id="TODO",
        name="TODO: LLM resource-exhaustion / denial of service",
        tactic="TODO",
        verified=False,
        notes="Placeholder category - find the correct ATLAS technique ID before use.",
    ),
    "training_data_extraction": AtlasTechnique(
        technique_id="TODO",
        name="TODO: memorized/training data extraction via prompting",
        tactic="TODO",
        verified=False,
        notes="Placeholder category - candidate IDs (e.g. around AML.T0024) not confirmed.",
    ),
    "multi_agent_lateral_propagation": AtlasTechnique(
        technique_id="TODO",
        name="TODO: injection propagating across a multi-agent pipeline",
        tactic="TODO",
        verified=False,
        notes="Placeholder category - needs lookup once payload logic is designed.",
    ),
}


def get_technique(category: str) -> AtlasTechnique:
    """Return the AtlasTechnique mapped to a payload category, raising if unknown."""
    try:
        return TAXONOMY[category]
    except KeyError as exc:
        raise KeyError(f"Unknown payload category: {category!r}") from exc


def list_categories() -> List[str]:
    """Return all known payload category names."""
    return list(TAXONOMY.keys())


def verified_categories() -> List[str]:
    """Return payload categories that have a confirmed real ATLAS technique ID."""
    return [name for name, tech in TAXONOMY.items() if tech.verified]
