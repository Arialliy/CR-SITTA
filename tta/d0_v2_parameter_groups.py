"""Fine-grained, fail-closed BN-affine inventory for D0-v2 P3.

The D0-v1 semantic grouping remains sealed.  This versioned module refines
the inventory to individual encoder stages and individual NS-FPN list items,
as required by the P3 gradient-alignment diagnosis.  It does not select a
scientific candidate and it does not run an optimizer step.

``final_head`` is retained as an auditable architecture group, but the frozen
NS-FPN head contains no ``BatchNorm2d`` affine tensor.  It is therefore
explicitly ``structurally_ineligible``; callers may not turn an empty group
into a fake single-group adaptation candidate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal

from torch import nn

from tta.parameter_groups import (
    FROZEN_NSFPN_NAMED_MODULES_SHA256,
    GroupInventoryEntry,
    ParameterGroupError,
    SemanticGroupDefinition,
    build_parameter_group_inventory,
)


D0_V2_FINE_GROUP_PROTOCOL_ID = "cr-sitta-d0-v2-fine-bn-affine-groups"
PARAMETER_KIND = "batchnorm2d_affine_only"
ELIGIBLE = "eligible"
STRUCTURALLY_INELIGIBLE = "structurally_ineligible"
FINAL_HEAD_REASON = "frozen_nsfpn_final_head_has_no_batchnorm2d_affine_parameters"

# Filled from the canonical current MSHNet_NSFPN inventory.  The exact value is
# patched only when this new v2 module is versioned; no v1 seal is involved.
FROZEN_D0_V2_FINE_INVENTORY_SHA256 = (
    "47b0aca39e4fa471d49cbb89513e1ec66f29ac3d8a258f4e1c63542dea6087c1"
)


class D0V2ParameterGroupError(ParameterGroupError):
    """The model or requested group violates the D0-v2 fine partition."""


FINE_GRAINED_GROUP_DEFINITIONS = (
    SemanticGroupDefinition("encoder_0", ("encoder_0",)),
    SemanticGroupDefinition("encoder_1", ("encoder_1",)),
    SemanticGroupDefinition("encoder_2", ("encoder_2",)),
    SemanticGroupDefinition("encoder_3", ("encoder_3",)),
    SemanticGroupDefinition("middle", ("middle_layer",)),
    *tuple(
        SemanticGroupDefinition(
            f"fpn_lateral_{index}", (f"fpn.lateral_convs.{index}",)
        )
        for index in range(4)
    ),
    *tuple(
        SemanticGroupDefinition(
            f"fpn_output_{index}", (f"fpn.fpn_convs.{index}",)
        )
        for index in range(4)
    ),
    *tuple(
        SemanticGroupDefinition(
            f"fpn_sfs_{index}", (f"fpn.crossattn_list.{index}",)
        )
        for index in range(3)
    ),
    SemanticGroupDefinition("decoder_3", ("decoder_3",)),
    SemanticGroupDefinition("decoder_2", ("decoder_2",)),
    SemanticGroupDefinition("decoder_1", ("decoder_1",)),
    SemanticGroupDefinition("decoder_0", ("decoder_0",)),
    SemanticGroupDefinition(
        "final_head",
        ("output_0", "output_1", "output_2", "output_3", "final"),
        allow_zero_bn_members=True,
    ),
)

FINE_GRAINED_GROUP_IDS = tuple(
    definition.group_id for definition in FINE_GRAINED_GROUP_DEFINITIONS
)
ELIGIBLE_GROUP_IDS = FINE_GRAINED_GROUP_IDS[:-1]


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class D0V2FineGroupInventoryEntry:
    group_id: str
    include_prefixes: tuple[str, ...]
    eligibility: Literal["eligible", "structurally_ineligible"]
    ineligibility_reason: str | None
    bn_module_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    scalar_parameter_count: int

    @property
    def bn_module_count(self) -> int:
        return len(self.bn_module_names)

    @property
    def parameter_tensor_count(self) -> int:
        return len(self.parameter_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "include_prefixes": list(self.include_prefixes),
            "eligibility": self.eligibility,
            "ineligibility_reason": self.ineligibility_reason,
            "bn_module_count": self.bn_module_count,
            "parameter_tensor_count": self.parameter_tensor_count,
            "scalar_parameter_count": self.scalar_parameter_count,
            "bn_module_names": list(self.bn_module_names),
            "parameter_names": list(self.parameter_names),
        }


@dataclass(frozen=True)
class D0V2FineGroupInventory:
    model_type: str
    named_modules_sha256: str
    fine_inventory_sha256: str
    groups: tuple[D0V2FineGroupInventoryEntry, ...]

    @property
    def eligible_group_count(self) -> int:
        return sum(group.eligibility == ELIGIBLE for group in self.groups)

    @property
    def structurally_ineligible_group_count(self) -> int:
        return sum(
            group.eligibility == STRUCTURALLY_INELIGIBLE for group in self.groups
        )

    @property
    def bn_module_count(self) -> int:
        return sum(group.bn_module_count for group in self.groups)

    @property
    def parameter_tensor_count(self) -> int:
        return sum(group.parameter_tensor_count for group in self.groups)

    @property
    def scalar_parameter_count(self) -> int:
        return sum(group.scalar_parameter_count for group in self.groups)

    def group(self, group_id: str) -> D0V2FineGroupInventoryEntry:
        matches = tuple(group for group in self.groups if group.group_id == group_id)
        if len(matches) != 1:
            raise D0V2ParameterGroupError(
                f"inventory has no unique group {group_id!r}"
            )
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "protocol_id": D0_V2_FINE_GROUP_PROTOCOL_ID,
            "parameter_kind": PARAMETER_KIND,
            "model_type": self.model_type,
            "named_modules_sha256": self.named_modules_sha256,
            "fine_inventory_sha256": self.fine_inventory_sha256,
            "totals": {
                "group_count": len(self.groups),
                "eligible_group_count": self.eligible_group_count,
                "structurally_ineligible_group_count": (
                    self.structurally_ineligible_group_count
                ),
                "bn_module_count": self.bn_module_count,
                "parameter_tensor_count": self.parameter_tensor_count,
                "scalar_parameter_count": self.scalar_parameter_count,
            },
            "groups": [group.to_dict() for group in self.groups],
            "authorization": {
                "scientific_gate_status": "unresolved",
                "scientific_selection_performed": False,
                "stage2_authorized": False,
            },
        }


def _fine_descriptor(
    *,
    model_type: str,
    named_modules_sha256: str,
    groups: Sequence[D0V2FineGroupInventoryEntry],
) -> dict[str, Any]:
    return {
        "protocol_id": D0_V2_FINE_GROUP_PROTOCOL_ID,
        "parameter_kind": PARAMETER_KIND,
        "model_type": model_type,
        "named_modules_sha256": named_modules_sha256,
        "groups": [group.to_dict() for group in groups],
    }


def _convert_entry(entry: GroupInventoryEntry) -> D0V2FineGroupInventoryEntry:
    if entry.group_id == "final_head":
        if (
            entry.bn_module_count != 0
            or entry.parameter_tensor_count != 0
            or entry.scalar_parameter_count != 0
        ):
            raise D0V2ParameterGroupError(
                "final_head is frozen as structurally_ineligible and must contain "
                "zero BatchNorm2d affine parameters"
            )
        eligibility: Literal["eligible", "structurally_ineligible"] = (
            STRUCTURALLY_INELIGIBLE
        )
        reason: str | None = FINAL_HEAD_REASON
    else:
        if entry.bn_module_count <= 0 or entry.parameter_tensor_count <= 0:
            raise D0V2ParameterGroupError(
                f"eligible group {entry.group_id!r} contains no BN affine tensors"
            )
        eligibility = ELIGIBLE
        reason = None
    return D0V2FineGroupInventoryEntry(
        group_id=entry.group_id,
        include_prefixes=entry.include_prefixes,
        eligibility=eligibility,
        ineligibility_reason=reason,
        bn_module_names=entry.bn_module_names,
        parameter_names=entry.parameter_names,
        scalar_parameter_count=entry.scalar_parameter_count,
    )


def build_d0_v2_fine_group_inventory(model: nn.Module) -> D0V2FineGroupInventory:
    """Build a complete, disjoint fine-grained inventory without selecting."""

    try:
        base = build_parameter_group_inventory(
            model, definitions=FINE_GRAINED_GROUP_DEFINITIONS
        )
    except ParameterGroupError as exc:
        raise D0V2ParameterGroupError(str(exc)) from exc
    groups = tuple(_convert_entry(entry) for entry in base.groups)
    if tuple(group.group_id for group in groups) != FINE_GRAINED_GROUP_IDS:
        raise D0V2ParameterGroupError("fine group order or membership drifted")
    descriptor = _fine_descriptor(
        model_type=base.model_type,
        named_modules_sha256=base.named_modules_sha256,
        groups=groups,
    )
    return D0V2FineGroupInventory(
        model_type=base.model_type,
        named_modules_sha256=base.named_modules_sha256,
        fine_inventory_sha256=_canonical_sha256(descriptor),
        groups=groups,
    )


def verify_frozen_d0_v2_fine_inventory(
    inventory: D0V2FineGroupInventory,
) -> None:
    """Bind the fine inventory to the frozen NS-FPN architecture."""

    if not isinstance(inventory, D0V2FineGroupInventory):
        raise D0V2ParameterGroupError(
            "inventory must be D0V2FineGroupInventory"
        )
    if inventory.named_modules_sha256 != FROZEN_NSFPN_NAMED_MODULES_SHA256:
        raise D0V2ParameterGroupError(
            "model.named_modules() drifted from the frozen NS-FPN architecture"
        )
    if inventory.fine_inventory_sha256 != FROZEN_D0_V2_FINE_INVENTORY_SHA256:
        raise D0V2ParameterGroupError(
            "D0-v2 fine BN-affine inventory drifted from frozen NS-FPN"
        )
    if (
        inventory.eligible_group_count != 20
        or inventory.structurally_ineligible_group_count != 1
        or inventory.bn_module_count != 53
        or inventory.parameter_tensor_count != 106
        or inventory.scalar_parameter_count != 8736
    ):
        raise D0V2ParameterGroupError("frozen D0-v2 fine inventory totals drifted")


def collect_d0_v2_single_group_parameters(
    model: nn.Module,
    *,
    group_id: str,
    require_frozen_nsfpn: bool = True,
) -> tuple[list[nn.Parameter], list[str]]:
    """Collect exactly one eligible fine group after full-model validation."""

    if not isinstance(group_id, str) or not group_id:
        raise D0V2ParameterGroupError("group_id must be a non-empty string")
    inventory = build_d0_v2_fine_group_inventory(model)
    if require_frozen_nsfpn:
        verify_frozen_d0_v2_fine_inventory(inventory)
    group = inventory.group(group_id)
    if group.eligibility == STRUCTURALLY_INELIGIBLE:
        raise D0V2ParameterGroupError(
            f"group {group_id!r} is structurally_ineligible: "
            f"{group.ineligibility_reason}"
        )
    if group.eligibility != ELIGIBLE or not group.parameter_names:
        raise D0V2ParameterGroupError(
            f"group {group_id!r} is not an eligible non-empty group"
        )

    named_parameters = dict(model.named_parameters())
    missing = [name for name in group.parameter_names if name not in named_parameters]
    if missing:
        raise D0V2ParameterGroupError(
            f"fine-group parameters disappeared from model: {missing}"
        )
    parameters = [named_parameters[name] for name in group.parameter_names]
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise D0V2ParameterGroupError(
            f"group {group_id!r} contains shared parameter objects"
        )
    if any(not name.endswith((".weight", ".bias")) for name in group.parameter_names):
        raise D0V2ParameterGroupError(
            f"group {group_id!r} contains a non-affine parameter name"
        )
    return parameters, list(group.parameter_names)


__all__ = [
    "D0V2FineGroupInventory",
    "D0V2FineGroupInventoryEntry",
    "D0V2ParameterGroupError",
    "D0_V2_FINE_GROUP_PROTOCOL_ID",
    "ELIGIBLE",
    "ELIGIBLE_GROUP_IDS",
    "FINAL_HEAD_REASON",
    "FINE_GRAINED_GROUP_DEFINITIONS",
    "FINE_GRAINED_GROUP_IDS",
    "FROZEN_D0_V2_FINE_INVENTORY_SHA256",
    "PARAMETER_KIND",
    "STRUCTURALLY_INELIGIBLE",
    "build_d0_v2_fine_group_inventory",
    "collect_d0_v2_single_group_parameters",
    "verify_frozen_d0_v2_fine_inventory",
]
