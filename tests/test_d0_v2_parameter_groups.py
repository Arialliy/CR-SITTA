from __future__ import annotations

import copy
import json

import pytest
from torch import nn

import test_source
from tta.d0_v2_parameter_groups import (
    D0V2ParameterGroupError,
    ELIGIBLE,
    ELIGIBLE_GROUP_IDS,
    FINAL_HEAD_REASON,
    FINE_GRAINED_GROUP_IDS,
    FROZEN_D0_V2_FINE_INVENTORY_SHA256,
    STRUCTURALLY_INELIGIBLE,
    build_d0_v2_fine_group_inventory,
    collect_d0_v2_single_group_parameters,
    verify_frozen_d0_v2_fine_inventory,
)


class _ToyFPN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            nn.Sequential(nn.Conv2d(1, 1, 1), nn.BatchNorm2d(1))
            for _ in range(4)
        )
        self.fpn_convs = nn.ModuleList(
            nn.Sequential(nn.Conv2d(1, 1, 1), nn.BatchNorm2d(1))
            for _ in range(4)
        )
        self.crossattn_list = nn.ModuleList(
            nn.Sequential(nn.Conv2d(1, 1, 1), nn.BatchNorm2d(1))
            for _ in range(3)
        )


class _ToyFineModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for name in (
            "encoder_0",
            "encoder_1",
            "encoder_2",
            "encoder_3",
            "middle_layer",
            "decoder_3",
            "decoder_2",
            "decoder_1",
            "decoder_0",
        ):
            setattr(self, name, nn.Sequential(nn.BatchNorm2d(1)))
        self.fpn = _ToyFPN()
        self.output_0 = nn.Conv2d(1, 1, 1)
        self.output_1 = nn.Conv2d(1, 1, 1)
        self.output_2 = nn.Conv2d(1, 1, 1)
        self.output_3 = nn.Conv2d(1, 1, 1)
        self.final = nn.Conv2d(1, 1, 1)


def test_frozen_nsfpn_has_exact_fine_inventory_and_structural_head_status() -> None:
    model = test_source.build_nsfpn_model()
    inventory = build_d0_v2_fine_group_inventory(model)
    verify_frozen_d0_v2_fine_inventory(inventory)

    assert tuple(group.group_id for group in inventory.groups) == (
        FINE_GRAINED_GROUP_IDS
    )
    assert inventory.fine_inventory_sha256 == (
        FROZEN_D0_V2_FINE_INVENTORY_SHA256
    )
    assert (
        inventory.eligible_group_count,
        inventory.structurally_ineligible_group_count,
        inventory.bn_module_count,
        inventory.parameter_tensor_count,
        inventory.scalar_parameter_count,
    ) == (20, 1, 53, 106, 8736)

    expected_counts = {
        "encoder_0": (2, 4, 64),
        "encoder_1": (5, 10, 320),
        "encoder_2": (5, 10, 640),
        "encoder_3": (5, 10, 1280),
        "middle": (5, 10, 2560),
        **{f"fpn_lateral_{index}": (1, 2, 128) for index in range(4)},
        **{f"fpn_output_{index}": (1, 2, 128) for index in range(4)},
        **{f"fpn_sfs_{index}": (2, 4, 256) for index in range(3)},
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
    } == expected_counts

    assert all(
        inventory.group(group_id).eligibility == ELIGIBLE
        for group_id in ELIGIBLE_GROUP_IDS
    )
    head = inventory.group("final_head")
    assert head.eligibility == STRUCTURALLY_INELIGIBLE
    assert head.ineligibility_reason == FINAL_HEAD_REASON
    assert head.parameter_names == ()
    assert inventory.to_dict()["authorization"] == {
        "scientific_gate_status": "unresolved",
        "scientific_selection_performed": False,
        "stage2_authorized": False,
    }
    json.dumps(inventory.to_dict(), allow_nan=False)


def test_single_group_collection_is_exact_and_rejects_final_head() -> None:
    model = test_source.build_nsfpn_model()
    parameters, names = collect_d0_v2_single_group_parameters(
        model, group_id="fpn_sfs_1"
    )
    assert names == [
        "fpn.crossattn_list.1.query_Conv.1.weight",
        "fpn.crossattn_list.1.query_Conv.1.bias",
        "fpn.crossattn_list.1.key_Conv.1.weight",
        "fpn.crossattn_list.1.key_Conv.1.bias",
    ]
    assert parameters == [dict(model.named_parameters())[name] for name in names]

    with pytest.raises(D0V2ParameterGroupError, match="structurally_ineligible"):
        collect_d0_v2_single_group_parameters(model, group_id="final_head")
    with pytest.raises(D0V2ParameterGroupError, match="no unique group"):
        collect_d0_v2_single_group_parameters(model, group_id="unknown")


def test_synthetic_model_supports_cpu_inventory_without_frozen_model_claim() -> None:
    model = _ToyFineModel()
    inventory = build_d0_v2_fine_group_inventory(model)
    assert inventory.eligible_group_count == 20
    assert inventory.structurally_ineligible_group_count == 1
    assert inventory.bn_module_count == 20

    parameters, names = collect_d0_v2_single_group_parameters(
        model,
        group_id="decoder_0",
        require_frozen_nsfpn=False,
    )
    assert len(parameters) == len(names) == 2
    with pytest.raises(
        D0V2ParameterGroupError, match="named_modules.*drifted"
    ):
        collect_d0_v2_single_group_parameters(
            model,
            group_id="decoder_0",
            require_frozen_nsfpn=True,
        )


def test_unclassified_bn_and_bn_in_structurally_ineligible_head_fail_closed() -> None:
    unclassified = _ToyFineModel()
    unclassified.extra_branch = nn.BatchNorm2d(1)
    with pytest.raises(D0V2ParameterGroupError, match="unclassified"):
        build_d0_v2_fine_group_inventory(unclassified)

    invalid_head = _ToyFineModel()
    invalid_head.final = nn.Sequential(nn.Conv2d(1, 1, 1), nn.BatchNorm2d(1))
    with pytest.raises(D0V2ParameterGroupError, match="structurally_ineligible"):
        build_d0_v2_fine_group_inventory(invalid_head)


def test_missing_fine_anchor_and_non_affine_bn_fail_closed() -> None:
    missing = _ToyFineModel()
    del missing.fpn.crossattn_list[2]
    with pytest.raises(D0V2ParameterGroupError, match="missing module anchors"):
        build_d0_v2_fine_group_inventory(missing)

    non_affine = _ToyFineModel()
    non_affine.decoder_0 = nn.Sequential(nn.BatchNorm2d(1, affine=False))
    with pytest.raises(D0V2ParameterGroupError, match="must have weight and bias"):
        build_d0_v2_fine_group_inventory(non_affine)


def test_inventory_is_immutable_and_model_is_not_mutated() -> None:
    model = _ToyFineModel()
    before = copy.deepcopy(model.state_dict())
    inventory = build_d0_v2_fine_group_inventory(model)
    assert all(
        before[name].equal(value) for name, value in model.state_dict().items()
    )
    with pytest.raises(Exception):
        inventory.groups[0].parameter_names += ("unsafe",)  # type: ignore[misc]
