"""Tests for the MITRE ATLAS payload taxonomy mapping."""
import pytest

from promptraid.payloads.taxonomy import (
    TAXONOMY,
    get_technique,
    list_categories,
    verified_categories,
)


def test_list_categories_matches_taxonomy_keys():
    assert set(list_categories()) == set(TAXONOMY.keys())


def test_at_least_five_verified_categories():
    assert len(verified_categories()) >= 5


def test_get_technique_returns_real_atlas_id_for_direct_injection():
    tech = get_technique("direct_instruction_override")
    assert tech.technique_id == "AML.T0051.000"
    assert tech.verified is True


def test_get_technique_returns_real_atlas_id_for_indirect_injection():
    tech = get_technique("indirect_tool_output_injection")
    assert tech.technique_id == "AML.T0051.001"


def test_get_technique_unknown_category_raises_keyerror():
    with pytest.raises(KeyError):
        get_technique("not_a_real_category")


def test_unverified_categories_are_flagged_todo():
    for name, tech in TAXONOMY.items():
        if not tech.verified:
            assert tech.technique_id == "TODO"


def test_all_verified_technique_ids_follow_atlas_format():
    for name, tech in TAXONOMY.items():
        if tech.verified:
            assert tech.technique_id.startswith("AML.T")


def test_no_duplicate_technique_ids_among_verified_categories():
    verified_ids = [TAXONOMY[name].technique_id for name in verified_categories()]
    assert len(verified_ids) == len(set(verified_ids))
