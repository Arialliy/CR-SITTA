from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path

import pytest
from torch import nn
import yaml

import test_source
from tta.parameter_groups import (
    AdaptableGroupSpec,
    DEFAULT_GROUP_ALIASES,
    DEFAULT_SEMANTIC_GROUPS,
    PARAMETER_GROUP_PROTOCOL_ID,
    PARAMETER_KIND,
    PILOT_GROUP_SPECS,
    ParameterGroupError,
    SemanticGroupDefinition,
    build_parameter_group_inventory,
    collect_adaptable_params,
    resolve_group_ids,
    validate_group_partition,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/parameter_group_pilot_v1.yaml"
CHECKPOINT_ROOT = ROOT / "results/retraining_fixed_split"
DATASETS = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
LOCAL_ARTIFACT_TEST_ENV = "NS_FPN_RUN_LOCAL_ARTIFACT_TESTS"
LOCAL_ARTIFACT_TESTS_ENABLED = os.environ.get(LOCAL_ARTIFACT_TEST_ENV) == "1"


@pytest.fixture(scope="module")
def nsfpn_model() -> nn.Module:
    return test_source.build_nsfpn_model()


def _all_bn_affine_names(model: nn.Module) -> list[str]:
    names: list[str] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.BatchNorm2d):
            continue
        assert module.affine and module.weight is not None and module.bias is not None
        names.extend((f"{module_name}.weight", f"{module_name}.bias"))
    return names


def test_current_nsfpn_partition_is_complete_disjoint_and_exact(
    nsfpn_model: nn.Module,
) -> None:
    inventory = build_parameter_group_inventory(nsfpn_model)

    assert inventory.protocol_id == PARAMETER_GROUP_PROTOCOL_ID
    assert inventory.parameter_kind == PARAMETER_KIND
    assert inventory.model_type == "model.MSHNet_NSFPN.MSHNet_NSFPN"
    assert inventory.named_modules_sha256 == (
        "581e0bc592ceea4e72420f54f9a20e9c9b07199b45928fa61ec3e60286b87990"
    )
    assert inventory.bn_affine_inventory_sha256 == (
        "3a4ba5e95c52100f9b1a75a277d33e4685c880b055df4f5a794ca03f9a060955"
    )
    assert (
        inventory.bn_module_count,
        inventory.parameter_tensor_count,
        inventory.scalar_parameter_count,
    ) == (53, 106, 8736)

    expected = {
        "encoder": (17, 34, 2304),
        "middle": (5, 10, 2560),
        "nsfpn_lateral": (4, 8, 512),
        "nsfpn_output": (4, 8, 512),
        "nsfpn_sfs": (6, 12, 768),
        "decoder_3": (4, 8, 1024),
        "decoder_2": (5, 10, 640),
        "decoder_1": (5, 10, 320),
        "decoder_0": (3, 6, 96),
        "final_head": (0, 0, 0),
    }
    assert {
        group.group_id: (
            group.bn_module_count,
            group.parameter_tensor_count,
            group.scalar_parameter_count,
        )
        for group in inventory.groups
    } == expected

    grouped_names = [
        name for group in inventory.groups for name in group.parameter_names
    ]
    assert grouped_names == _all_bn_affine_names(nsfpn_model)
    assert len(grouped_names) == len(set(grouped_names))
    assert inventory.group("final_head").allow_zero_bn_members is True
    assert inventory.group("final_head").bn_module_names == ()


def test_pilot_specs_collect_only_bn_affine_and_support_p0_to_p4(
    nsfpn_model: nn.Module,
) -> None:
    expected = {
        "P0": (106, 8736),
        "P1": (6, 96),
        "P2": (16, 416),
        "P3": (34, 2080),
        "P4": (42, 2592),
    }
    model_parameters = dict(nsfpn_model.named_parameters())
    for pilot_id, (tensor_count, scalar_count) in expected.items():
        parameters, names = collect_adaptable_params(
            nsfpn_model,
            group_spec=PILOT_GROUP_SPECS[pilot_id],
        )
        assert len(parameters) == len(names) == tensor_count
        assert sum(parameter.numel() for parameter in parameters) == scalar_count
        assert len({id(parameter) for parameter in parameters}) == tensor_count
        assert all(
            model_parameters[name] is parameter
            for name, parameter in zip(names, parameters)
        )
        assert all(name.endswith((".weight", ".bias")) for name in names)

    p0_names = collect_adaptable_params(
        nsfpn_model, group_spec=PILOT_GROUP_SPECS["P0"]
    )[1]
    assert p0_names == _all_bn_affine_names(nsfpn_model)


def test_final_head_is_explicitly_zero_member_and_not_a_fake_bn_pilot(
    nsfpn_model: nn.Module,
) -> None:
    inventory = build_parameter_group_inventory(nsfpn_model)
    final_head = inventory.group("final_head")
    assert final_head.include_prefixes == (
        "output_0",
        "output_1",
        "output_2",
        "output_3",
        "final",
    )
    assert final_head.bn_module_count == 0
    assert final_head.parameter_tensor_count == 0
    assert final_head.scalar_parameter_count == 0
    with pytest.raises(ParameterGroupError, match="zero BN affine tensors"):
        collect_adaptable_params(
            nsfpn_model,
            group_spec=PILOT_GROUP_SPECS["P5"],
        )


def test_unknown_missing_nonaffine_and_overlapping_groups_fail_closed(
    nsfpn_model: nn.Module,
) -> None:
    unknown_bn = deepcopy(nsfpn_model)
    unknown_bn.unreviewed_branch = nn.BatchNorm2d(1)
    with pytest.raises(ParameterGroupError, match="unclassified"):
        build_parameter_group_inventory(unknown_bn)

    nonaffine = deepcopy(nsfpn_model)
    nonaffine.decoder_0[0].bn1 = nn.BatchNorm2d(16, affine=False)
    with pytest.raises(ParameterGroupError, match="must have weight and bias"):
        build_parameter_group_inventory(nonaffine)

    recognized_prefix_but_missing_bn = deepcopy(nsfpn_model)
    recognized_prefix_but_missing_bn.decoder_0[0].bn1 = nn.Identity()
    with pytest.raises(ParameterGroupError, match="inventory drifted"):
        collect_adaptable_params(
            recognized_prefix_but_missing_bn,
            group_spec=PILOT_GROUP_SPECS["P1"],
        )

    missing_anchor_definitions = tuple(
        SemanticGroupDefinition(
            value.group_id,
            ("renamed_decoder_0",)
            if value.group_id == "decoder_0"
            else value.include_prefixes,
            value.allow_zero_bn_members,
            value.description,
        )
        for value in DEFAULT_SEMANTIC_GROUPS
    )
    with pytest.raises(ParameterGroupError, match="missing module anchors"):
        build_parameter_group_inventory(
            nsfpn_model,
            definitions=missing_anchor_definitions,
        )

    overlapping = (
        *DEFAULT_SEMANTIC_GROUPS,
        SemanticGroupDefinition("bad_overlap", ("decoder_0.0",)),
    )
    with pytest.raises(ParameterGroupError, match="prefixes overlap"):
        validate_group_partition(overlapping)


def test_unknown_or_overlapping_alias_selection_fails_closed() -> None:
    with pytest.raises(ParameterGroupError, match="unknown semantic group"):
        resolve_group_ids(AdaptableGroupSpec(("unknown_group",)))
    with pytest.raises(ParameterGroupError, match="overlap after alias expansion"):
        resolve_group_ids(AdaptableGroupSpec(("decoder_low", "decoder_0")))

    assert resolve_group_ids(PILOT_GROUP_SPECS["P2"]) == (
        "decoder_1",
        "decoder_0",
    )
    assert resolve_group_ids(PILOT_GROUP_SPECS["P4"]) == (
        "nsfpn_output",
        "decoder_3",
        "decoder_2",
        "decoder_1",
        "decoder_0",
    )
    with pytest.raises(TypeError):
        DEFAULT_GROUP_ALIASES["unsafe"] = ("encoder",)  # type: ignore[index]
    with pytest.raises(TypeError):
        PILOT_GROUP_SPECS["unsafe"] = AdaptableGroupSpec(("encoder",))  # type: ignore[index]


@pytest.mark.skipif(
    not LOCAL_ARTIFACT_TESTS_ENABLED,
    reason=(
        "requires ignored local result artifacts; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)
def test_three_frozen_best_miou_checkpoints_share_the_frozen_inventory() -> None:
    assert CONFIG_PATH.is_file() and not CONFIG_PATH.is_symlink()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    checkpoint_contract = config["frozen_host"]["best_miou_checkpoints"]
    expected_inventory = config["frozen_host"]["model_inventory"]
    checkpoints = {
        dataset: CHECKPOINT_ROOT / dataset / "best_miou.pth.tar"
        for dataset in DATASETS
    }
    assert all(
        checkpoint.is_file() and not checkpoint.is_symlink()
        for checkpoint in checkpoints.values()
    )

    for dataset, checkpoint in checkpoints.items():
        expected_checkpoint = checkpoint_contract[dataset]
        assert checkpoint.relative_to(ROOT).as_posix() == expected_checkpoint["path"]
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == (
            expected_checkpoint["sha256"]
        )
        model = test_source.build_nsfpn_model()
        assert test_source.load_trusted_checkpoint(model, checkpoint) == "state_dict"
        inventory = build_parameter_group_inventory(model)
        assert inventory.named_modules_sha256 == expected_inventory[
            "named_modules_sha256"
        ]
        assert inventory.bn_affine_inventory_sha256 == expected_inventory[
            "bn_affine_inventory_sha256"
        ]
        assert inventory.bn_module_count == expected_inventory["bn_module_count"]
        assert inventory.parameter_tensor_count == expected_inventory[
            "parameter_tensor_count"
        ]
        assert inventory.scalar_parameter_count == expected_inventory[
            "scalar_parameter_count"
        ]


def test_frozen_config_matches_code_prefixes_aliases_counts_and_hash() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["schema_version"] == 1
    assert config["protocol_id"] == PARAMETER_GROUP_PROTOCOL_ID
    assert config["parameter_space"]["kind"] == PARAMETER_KIND
    definitions = config["parameter_space"]["leaf_groups"]
    assert {
        group.group_id: list(group.include_prefixes)
        for group in DEFAULT_SEMANTIC_GROUPS
    } == {
        group_id: value["include_prefixes"]
        for group_id, value in definitions.items()
    }
    assert dict(DEFAULT_GROUP_ALIASES) == {
        key: tuple(value) for key, value in config["parameter_space"]["aliases"].items()
    }
    assert {
        pilot_id: list(spec.group_ids)
        for pilot_id, spec in PILOT_GROUP_SPECS.items()
    } == {
        pilot_id: value["select"]
        for pilot_id, value in config["factorial_pilot"].items()
    }
    source_path = ROOT / config["implementation"]["path"]
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == config[
        "implementation"
    ]["sha256"]
