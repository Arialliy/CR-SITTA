"""Fail-closed semantic parameter groups for the frozen NS-FPN host model.

The grouping boundary is deliberately narrower than a general parameter
selector: only affine ``torch.nn.BatchNorm2d`` weight/bias tensors are
adaptable.  Every such module in the model must belong to exactly one frozen
leaf group before *any* subset can be collected.  Architecture drift,
overlapping prefixes, missing group anchors, non-affine BN, and unclassified BN
therefore fail before an optimizer can be constructed.

The leaf groups are a non-overlapping partition.  Named aliases such as
``decoder_low`` and ``all_decoder`` are selection conveniences, not additional
overlapping groups.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any

from torch import nn


PARAMETER_GROUP_PROTOCOL_ID = "cr-sitta-nsfpn-bn-affine-groups-v1"
PARAMETER_KIND = "batchnorm2d_affine_only"
FROZEN_NSFPN_NAMED_MODULES_SHA256 = (
    "581e0bc592ceea4e72420f54f9a20e9c9b07199b45928fa61ec3e60286b87990"
)
FROZEN_NSFPN_BN_AFFINE_INVENTORY_SHA256 = (
    "3a4ba5e95c52100f9b1a75a277d33e4685c880b055df4f5a794ca03f9a060955"
)


class ParameterGroupError(ValueError):
    """Raised when a model cannot satisfy the frozen grouping contract."""


@dataclass(frozen=True)
class SemanticGroupDefinition:
    """One non-overlapping leaf in the semantic architecture partition."""

    group_id: str
    include_prefixes: tuple[str, ...]
    allow_zero_bn_members: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ParameterGroupError("group_id must be a non-empty string")
        if not isinstance(self.include_prefixes, tuple) or not self.include_prefixes:
            raise ParameterGroupError(
                f"group {self.group_id!r} must have a non-empty prefix tuple"
            )
        for prefix in self.include_prefixes:
            if (
                not isinstance(prefix, str)
                or not prefix
                or prefix.startswith(".")
                or prefix.endswith(".")
                or ".." in prefix
            ):
                raise ParameterGroupError(
                    f"group {self.group_id!r} has invalid module prefix {prefix!r}"
                )
        if len(set(self.include_prefixes)) != len(self.include_prefixes):
            raise ParameterGroupError(
                f"group {self.group_id!r} contains duplicate prefixes"
            )
        if not isinstance(self.allow_zero_bn_members, bool):
            raise ParameterGroupError("allow_zero_bn_members must be bool")


@dataclass(frozen=True)
class AdaptableGroupSpec:
    """A requested subset or alias composition for one adaptation candidate."""

    group_ids: tuple[str, ...] = ("all_bn",)
    parameter_kind: str = PARAMETER_KIND

    def __post_init__(self) -> None:
        if not isinstance(self.group_ids, tuple) or not self.group_ids:
            raise ParameterGroupError("group_ids must be a non-empty tuple")
        if any(not isinstance(value, str) or not value for value in self.group_ids):
            raise ParameterGroupError("every requested group ID must be non-empty")
        if len(set(self.group_ids)) != len(self.group_ids):
            raise ParameterGroupError("requested group IDs must be unique")
        if self.parameter_kind != PARAMETER_KIND:
            raise ParameterGroupError(
                f"unsupported parameter_kind {self.parameter_kind!r}; "
                f"expected {PARAMETER_KIND!r}"
            )


@dataclass(frozen=True)
class GroupInventoryEntry:
    group_id: str
    include_prefixes: tuple[str, ...]
    allow_zero_bn_members: bool
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
            "allow_zero_bn_members": self.allow_zero_bn_members,
            "bn_module_count": self.bn_module_count,
            "parameter_tensor_count": self.parameter_tensor_count,
            "scalar_parameter_count": self.scalar_parameter_count,
            "bn_module_names": list(self.bn_module_names),
            "parameter_names": list(self.parameter_names),
        }


@dataclass(frozen=True)
class ModelParameterGroupInventory:
    protocol_id: str
    parameter_kind: str
    model_type: str
    named_modules_sha256: str
    bn_affine_inventory_sha256: str
    groups: tuple[GroupInventoryEntry, ...]

    @property
    def bn_module_count(self) -> int:
        return sum(group.bn_module_count for group in self.groups)

    @property
    def parameter_tensor_count(self) -> int:
        return sum(group.parameter_tensor_count for group in self.groups)

    @property
    def scalar_parameter_count(self) -> int:
        return sum(group.scalar_parameter_count for group in self.groups)

    def group(self, group_id: str) -> GroupInventoryEntry:
        matches = [value for value in self.groups if value.group_id == group_id]
        if len(matches) != 1:
            raise ParameterGroupError(f"inventory has no unique group {group_id!r}")
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "protocol_id": self.protocol_id,
            "parameter_kind": self.parameter_kind,
            "model_type": self.model_type,
            "named_modules_sha256": self.named_modules_sha256,
            "bn_affine_inventory_sha256": self.bn_affine_inventory_sha256,
            "totals": {
                "bn_module_count": self.bn_module_count,
                "parameter_tensor_count": self.parameter_tensor_count,
                "scalar_parameter_count": self.scalar_parameter_count,
            },
            "groups": [group.to_dict() for group in self.groups],
        }


# These prefixes were read from MSHNet_NSFPN.named_modules() after strict loads
# of all three frozen best_miou checkpoints.  They are intentionally explicit:
# an encoder_4, a renamed FPN path, or a new BN-bearing branch must be reviewed
# and versioned instead of being silently absorbed.
DEFAULT_SEMANTIC_GROUPS = (
    SemanticGroupDefinition(
        "encoder",
        ("encoder_0", "encoder_1", "encoder_2", "encoder_3"),
        description="four encoder stages",
    ),
    SemanticGroupDefinition(
        "middle",
        ("middle_layer",),
        description="bottleneck residual stage",
    ),
    SemanticGroupDefinition(
        "nsfpn_lateral",
        ("fpn.lateral_convs",),
        description="NS-FPN lateral projection convolutions",
    ),
    SemanticGroupDefinition(
        "nsfpn_output",
        ("fpn.fpn_convs",),
        description="NS-FPN output-side convolutions",
    ),
    SemanticGroupDefinition(
        "nsfpn_sfs",
        ("fpn.crossattn_list",),
        description="SFS query/key convolutional BN",
    ),
    SemanticGroupDefinition(
        "decoder_3",
        ("decoder_3",),
        description="highest/coarsest decoder stage",
    ),
    SemanticGroupDefinition(
        "decoder_2",
        ("decoder_2",),
        description="decoder stage 2",
    ),
    SemanticGroupDefinition(
        "decoder_1",
        ("decoder_1",),
        description="decoder stage 1",
    ),
    SemanticGroupDefinition(
        "decoder_0",
        ("decoder_0",),
        description="final/full-resolution decoder stage",
    ),
    SemanticGroupDefinition(
        "final_head",
        ("output_0", "output_1", "output_2", "output_3", "final"),
        allow_zero_bn_members=True,
        description="convolutional prediction heads; current model has no BN",
    ),
)


# Aliases compose leaf groups and are never inventory partitions themselves.
# This preserves non-overlap while directly expressing the P0--P5 pilots.
DEFAULT_GROUP_ALIASES: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "decoder_high": ("decoder_3", "decoder_2"),
    "decoder_low": ("decoder_1", "decoder_0"),
    "all_decoder": ("decoder_3", "decoder_2", "decoder_1", "decoder_0"),
    "nsfpn_output_plus_decoder": (
        "nsfpn_output",
        "decoder_3",
        "decoder_2",
        "decoder_1",
        "decoder_0",
    ),
    "all_bn": tuple(group.group_id for group in DEFAULT_SEMANTIC_GROUPS),
})


PILOT_GROUP_SPECS: Mapping[str, AdaptableGroupSpec] = MappingProxyType({
    "P0": AdaptableGroupSpec(("all_bn",)),
    "P1": AdaptableGroupSpec(("decoder_0",)),
    "P2": AdaptableGroupSpec(("decoder_low",)),
    "P3": AdaptableGroupSpec(("all_decoder",)),
    "P4": AdaptableGroupSpec(("nsfpn_output_plus_decoder",)),
    "P5": AdaptableGroupSpec(("final_head",)),
})


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _type_name(module: nn.Module) -> str:
    cls = type(module)
    return f"{cls.__module__}.{cls.__qualname__}"


def _prefix_matches(module_name: str, prefix: str) -> bool:
    return module_name == prefix or module_name.startswith(f"{prefix}.")


def _prefixes_overlap(left: str, right: str) -> bool:
    return _prefix_matches(left, right) or _prefix_matches(right, left)


def validate_group_partition(
    definitions: Sequence[SemanticGroupDefinition],
) -> tuple[SemanticGroupDefinition, ...]:
    """Validate a static, non-overlapping semantic leaf partition."""

    if isinstance(definitions, (str, bytes)):
        raise ParameterGroupError("group definitions must be a sequence")
    values = tuple(definitions)
    if not values:
        raise ParameterGroupError("group definitions cannot be empty")
    if any(not isinstance(value, SemanticGroupDefinition) for value in values):
        raise ParameterGroupError(
            "group definitions must contain SemanticGroupDefinition values"
        )
    group_ids = [value.group_id for value in values]
    if len(set(group_ids)) != len(group_ids):
        raise ParameterGroupError("semantic group IDs must be unique")
    flattened = [
        (definition.group_id, prefix)
        for definition in values
        for prefix in definition.include_prefixes
    ]
    for index, (left_group, left_prefix) in enumerate(flattened):
        for right_group, right_prefix in flattened[index + 1 :]:
            if _prefixes_overlap(left_prefix, right_prefix):
                raise ParameterGroupError(
                    "semantic prefixes overlap: "
                    f"{left_group}:{left_prefix!r} and "
                    f"{right_group}:{right_prefix!r}"
                )
    return values


def _validate_aliases(
    definitions: Sequence[SemanticGroupDefinition],
    aliases: Mapping[str, tuple[str, ...]],
) -> None:
    leaf_ids = {value.group_id for value in definitions}
    if not isinstance(aliases, Mapping):
        raise ParameterGroupError("aliases must be a mapping")
    if leaf_ids & set(aliases):
        raise ParameterGroupError("an alias cannot reuse a leaf group ID")
    for alias, targets in aliases.items():
        if not isinstance(alias, str) or not alias:
            raise ParameterGroupError("alias IDs must be non-empty strings")
        if not isinstance(targets, tuple) or not targets:
            raise ParameterGroupError(f"alias {alias!r} must have a non-empty tuple")
        if any(not isinstance(target, str) or not target for target in targets):
            raise ParameterGroupError(f"alias {alias!r} has an invalid target")

    def visit(value: str, stack: tuple[str, ...]) -> None:
        if value in leaf_ids:
            return
        if value not in aliases:
            raise ParameterGroupError(f"alias references unknown group {value!r}")
        if value in stack:
            raise ParameterGroupError(f"alias cycle detected: {(*stack, value)}")
        for target in aliases[value]:
            visit(target, (*stack, value))

    for alias in aliases:
        visit(alias, ())


def resolve_group_ids(
    group_spec: AdaptableGroupSpec,
    *,
    definitions: Sequence[SemanticGroupDefinition] = DEFAULT_SEMANTIC_GROUPS,
    aliases: Mapping[str, tuple[str, ...]] = DEFAULT_GROUP_ALIASES,
) -> tuple[str, ...]:
    """Resolve aliases to a canonical leaf order and reject overlap."""

    if not isinstance(group_spec, AdaptableGroupSpec):
        raise ParameterGroupError("group_spec must be AdaptableGroupSpec")
    values = validate_group_partition(definitions)
    _validate_aliases(values, aliases)
    leaf_ids = {value.group_id for value in values}
    expanded: list[str] = []

    def expand(value: str, stack: tuple[str, ...]) -> None:
        if value in leaf_ids:
            expanded.append(value)
            return
        if value not in aliases:
            raise ParameterGroupError(f"unknown semantic group or alias {value!r}")
        if value in stack:
            raise ParameterGroupError(f"alias cycle detected: {(*stack, value)}")
        for target in aliases[value]:
            expand(target, (*stack, value))

    for requested in group_spec.group_ids:
        expand(requested, ())
    duplicates = sorted({value for value in expanded if expanded.count(value) > 1})
    if duplicates:
        raise ParameterGroupError(
            "requested groups overlap after alias expansion; "
            f"duplicate_leaf_groups={duplicates}"
        )
    selected = set(expanded)
    return tuple(value.group_id for value in values if value.group_id in selected)


def build_parameter_group_inventory(
    model: nn.Module,
    *,
    definitions: Sequence[SemanticGroupDefinition] = DEFAULT_SEMANTIC_GROUPS,
) -> ModelParameterGroupInventory:
    """Inventory every adaptable BN affine tensor under an exact partition."""

    if not isinstance(model, nn.Module):
        raise ParameterGroupError("model must be torch.nn.Module")
    values = validate_group_partition(definitions)
    named_modules = tuple(model.named_modules())
    names = {name for name, _ in named_modules}
    if len(names) != len(named_modules):
        raise ParameterGroupError("model.named_modules() returned duplicate names")

    # Every frozen prefix must remain an actual module anchor.  A typo or model
    # rename cannot be hidden by an allowed-empty group.
    for definition in values:
        missing_anchors = [
            prefix for prefix in definition.include_prefixes if prefix not in names
        ]
        if missing_anchors:
            raise ParameterGroupError(
                f"group {definition.group_id!r} has missing module anchors: "
                f"{missing_anchors}"
            )

    mutable: dict[str, dict[str, Any]] = {
        value.group_id: {
            "modules": [],
            "parameters": [],
            "scalar_count": 0,
        }
        for value in values
    }
    seen_parameter_ids: set[int] = set()
    adaptable_parameter_names: list[str] = []
    bn_descriptors: list[dict[str, Any]] = []

    for module_name, module in named_modules:
        if not isinstance(module, nn.BatchNorm2d):
            continue
        matches = [
            definition
            for definition in values
            if any(
                _prefix_matches(module_name, prefix)
                for prefix in definition.include_prefixes
            )
        ]
        if len(matches) != 1:
            reason = "unclassified" if not matches else "multiply classified"
            raise ParameterGroupError(
                f"BatchNorm2d {module_name!r} is {reason}; "
                f"matching_groups={[value.group_id for value in matches]}"
            )
        if not module.affine or module.weight is None or module.bias is None:
            raise ParameterGroupError(
                f"BatchNorm2d {module_name!r} must have weight and bias affine tensors"
            )
        definition = matches[0]
        parameter_names = (f"{module_name}.weight", f"{module_name}.bias")
        for parameter, parameter_name in zip(
            (module.weight, module.bias), parameter_names
        ):
            if id(parameter) in seen_parameter_ids:
                raise ParameterGroupError(
                    f"adaptable BN affine tensor is shared or duplicated: {parameter_name}"
                )
            seen_parameter_ids.add(id(parameter))
            adaptable_parameter_names.append(parameter_name)
            mutable[definition.group_id]["parameters"].append(parameter_name)
            mutable[definition.group_id]["scalar_count"] += parameter.numel()
        mutable[definition.group_id]["modules"].append(module_name)
        bn_descriptors.append(
            {
                "group_id": definition.group_id,
                "module_name": module_name,
                "module_type": _type_name(module),
                "num_features": module.num_features,
                "eps": repr(module.eps),
                "momentum": repr(module.momentum),
                "track_running_stats": module.track_running_stats,
                "parameter_names": list(parameter_names),
                "parameter_shapes": [
                    list(module.weight.shape),
                    list(module.bias.shape),
                ],
            }
        )

    model_parameter_names = dict(model.named_parameters())
    missing_parameters = [
        name for name in adaptable_parameter_names if name not in model_parameter_names
    ]
    if missing_parameters:
        raise ParameterGroupError(
            f"BN affine parameters are absent from model.named_parameters(): "
            f"{missing_parameters}"
        )
    if len(set(adaptable_parameter_names)) != len(adaptable_parameter_names):
        raise ParameterGroupError("BN affine parameter names are not unique")

    groups: list[GroupInventoryEntry] = []
    for definition in values:
        group_values = mutable[definition.group_id]
        if not group_values["modules"] and not definition.allow_zero_bn_members:
            raise ParameterGroupError(
                f"required semantic group {definition.group_id!r} has zero BN members"
            )
        groups.append(
            GroupInventoryEntry(
                group_id=definition.group_id,
                include_prefixes=definition.include_prefixes,
                allow_zero_bn_members=definition.allow_zero_bn_members,
                bn_module_names=tuple(group_values["modules"]),
                parameter_names=tuple(group_values["parameters"]),
                scalar_parameter_count=group_values["scalar_count"],
            )
        )

    named_module_descriptor = [
        {"name": name, "type": _type_name(module)} for name, module in named_modules
    ]
    partition_descriptor = [
        {
            "group_id": value.group_id,
            "include_prefixes": list(value.include_prefixes),
            "allow_zero_bn_members": value.allow_zero_bn_members,
        }
        for value in values
    ]
    return ModelParameterGroupInventory(
        protocol_id=PARAMETER_GROUP_PROTOCOL_ID,
        parameter_kind=PARAMETER_KIND,
        model_type=_type_name(model),
        named_modules_sha256=_canonical_sha256(named_module_descriptor),
        bn_affine_inventory_sha256=_canonical_sha256(
            {
                "protocol_id": PARAMETER_GROUP_PROTOCOL_ID,
                "partition": partition_descriptor,
                "batchnorm2d_affine": bn_descriptors,
            }
        ),
        groups=tuple(groups),
    )


def verify_frozen_nsfpn_inventory(
    inventory: ModelParameterGroupInventory,
    *,
    expected_named_modules_sha256: str = FROZEN_NSFPN_NAMED_MODULES_SHA256,
    expected_bn_affine_inventory_sha256: str = (
        FROZEN_NSFPN_BN_AFFINE_INVENTORY_SHA256
    ),
) -> None:
    """Bind a computed inventory to the three-checkpoint NS-FPN architecture."""

    if not isinstance(inventory, ModelParameterGroupInventory):
        raise ParameterGroupError("inventory must be ModelParameterGroupInventory")
    for value, label in (
        (expected_named_modules_sha256, "expected_named_modules_sha256"),
        (
            expected_bn_affine_inventory_sha256,
            "expected_bn_affine_inventory_sha256",
        ),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ParameterGroupError(f"{label} must be lowercase 64-hex SHA256")
    if inventory.named_modules_sha256 != expected_named_modules_sha256:
        raise ParameterGroupError(
            "model.named_modules() inventory drifted from the three frozen "
            "best_miou checkpoints: "
            f"expected={expected_named_modules_sha256}, "
            f"observed={inventory.named_modules_sha256}"
        )
    if inventory.bn_affine_inventory_sha256 != expected_bn_affine_inventory_sha256:
        raise ParameterGroupError(
            "BN-affine semantic inventory drifted from the three frozen "
            "best_miou checkpoints: "
            f"expected={expected_bn_affine_inventory_sha256}, "
            f"observed={inventory.bn_affine_inventory_sha256}"
        )


def collect_adaptable_params(
    model: nn.Module,
    *,
    group_spec: AdaptableGroupSpec,
    definitions: Sequence[SemanticGroupDefinition] = DEFAULT_SEMANTIC_GROUPS,
    aliases: Mapping[str, tuple[str, ...]] = DEFAULT_GROUP_ALIASES,
    expected_named_modules_sha256: str | None = FROZEN_NSFPN_NAMED_MODULES_SHA256,
    expected_bn_affine_inventory_sha256: str | None = (
        FROZEN_NSFPN_BN_AFFINE_INVENTORY_SHA256
    ),
) -> tuple[list[nn.Parameter], list[str]]:
    """Collect a non-empty BN-affine subset after validating full coverage.

    Full-model inventory is always validated first, even for a one-group
    request.  This prevents a newly added, unclassified BN from being ignored
    merely because the caller selected a different late-decoder group.
    """

    inventory = build_parameter_group_inventory(model, definitions=definitions)
    if (expected_named_modules_sha256 is None) != (
        expected_bn_affine_inventory_sha256 is None
    ):
        raise ParameterGroupError(
            "both frozen inventory hashes must be supplied, or both explicitly None"
        )
    if expected_named_modules_sha256 is not None:
        assert expected_bn_affine_inventory_sha256 is not None
        verify_frozen_nsfpn_inventory(
            inventory,
            expected_named_modules_sha256=expected_named_modules_sha256,
            expected_bn_affine_inventory_sha256=(
                expected_bn_affine_inventory_sha256
            ),
        )
    selected_ids = resolve_group_ids(
        group_spec,
        definitions=definitions,
        aliases=aliases,
    )
    selected_names = [
        parameter_name
        for group in inventory.groups
        if group.group_id in selected_ids
        for parameter_name in group.parameter_names
    ]
    if not selected_names:
        raise ParameterGroupError(
            "requested adaptation selection contains zero BN affine tensors; "
            f"groups={group_spec.group_ids}. The current final_head has no BN "
            "and is not a runnable BN-affine pilot."
        )
    named_parameters = dict(model.named_parameters())
    try:
        parameters = [named_parameters[name] for name in selected_names]
    except KeyError as exc:
        raise ParameterGroupError(
            f"inventory parameter disappeared from named_parameters(): {exc}"
        ) from exc
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ParameterGroupError("selected adaptable parameters overlap")
    return parameters, selected_names


__all__ = [
    "AdaptableGroupSpec",
    "DEFAULT_GROUP_ALIASES",
    "DEFAULT_SEMANTIC_GROUPS",
    "FROZEN_NSFPN_BN_AFFINE_INVENTORY_SHA256",
    "FROZEN_NSFPN_NAMED_MODULES_SHA256",
    "GroupInventoryEntry",
    "ModelParameterGroupInventory",
    "PARAMETER_GROUP_PROTOCOL_ID",
    "PARAMETER_KIND",
    "PILOT_GROUP_SPECS",
    "ParameterGroupError",
    "SemanticGroupDefinition",
    "build_parameter_group_inventory",
    "collect_adaptable_params",
    "resolve_group_ids",
    "validate_group_partition",
    "verify_frozen_nsfpn_inventory",
]
