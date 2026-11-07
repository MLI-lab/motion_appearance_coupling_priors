"""HP sharing across pretrained models: tune once on a reference model, reuse its tuned HPs.

Models agreeing on every swept key not listed in ``<name>_hp_share_over[<stage>]`` form a group; only its reference is tuned.
"""
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.prefect.sweeping import get_hydra_sweep_combos

Label = Tuple[str, Any]


def _combo_len(stage_sweep: Mapping[str, Any]) -> int:
    # every combo of a stage has the same (expanded, '&'-split) keys, so the first one tells
    return len(get_hydra_sweep_combos(dict(stage_sweep))[0])


def split_labels_by_stage(
    labels: Sequence[Label], stage_sweeps: Mapping[str, Mapping[str, Any]]
) -> Dict[str, List[Label]]:
    """Split a model's flat label list (stage combos concatenated in ``stage_sweeps`` order)."""
    expected = sum(_combo_len(s) for s in stage_sweeps.values())
    if len(labels) != expected:
        raise ValueError(
            f"hp sharing: model labels {list(labels)} have {len(labels)} entries, but the stage "
            f"sweeps {list(stage_sweeps)} produce {expected}; cannot attribute keys to stages"
        )
    out: Dict[str, List[Label]] = {}
    pos = 0
    for stage, sweep in stage_sweeps.items():
        n = _combo_len(sweep)
        out[stage] = list(labels[pos : pos + n])
        pos += n
    return out


def _validate(
    share_over: Mapping[str, Sequence[str]],
    reference: Optional[Mapping[str, Mapping[str, Any]]],
    stage_sweeps: Mapping[str, Mapping[str, Any]],
    name: str,
) -> None:
    for stage, keys in share_over.items():
        if stage not in stage_sweeps:
            raise ValueError(f"{name}_hp_share_over: unknown stage '{stage}', known stages: {list(stage_sweeps)}")
        swept = [k for k, _ in get_hydra_sweep_combos(dict(stage_sweeps[stage]))[0]]
        for key in keys:
            if key not in swept:
                raise ValueError(
                    f"{name}_hp_share_over.{stage}: '{key}' is not swept in {stage}_task_overrides "
                    f"(swept keys: {swept})"
                )
    if reference is None:
        return
    if set(reference) != set(share_over):
        raise ValueError(
            f"{name}_hp_share_reference stages {sorted(reference)} must equal "
            f"{name}_hp_share_over stages {sorted(share_over)}"
        )
    for stage, keys in share_over.items():
        ref_keys = set(reference[stage])
        if ref_keys != set(keys):
            raise ValueError(
                f"{name}_hp_share_reference.{stage} must give a value for exactly the shared keys; "
                f"missing: {sorted(set(keys) - ref_keys)}, extra: {sorted(ref_keys - set(keys))}"
            )
        combos = get_hydra_sweep_combos(dict(stage_sweeps[stage]))
        for key, value in reference[stage].items():
            values = []
            for combo in combos:
                values += [v for k, v in combo if k == key and v not in values]
            if value not in values:
                raise ValueError(
                    f"{name}_hp_share_reference.{stage}.{key}={value!r} is not one of its sweep values {values}"
                )


def resolve_hp_share_references(
    model_labels: Sequence[Sequence[Label]],
    stage_sweeps: Mapping[str, Mapping[str, Any]],
    share_over: Optional[Mapping[str, Sequence[str]]],
    reference: Optional[Mapping[str, Mapping[str, Any]]],
    name: str,
    logger,
) -> List[int]:
    """Return, per model, the index of the model whose tuned HPs it uses (itself if tuned)."""
    if not share_over:
        if reference:
            raise ValueError(f"{name}_hp_share_reference is set but {name}_hp_share_over is empty")
        return list(range(len(model_labels)))
    _validate(share_over, reference, stage_sweeps, name)

    shared = {(stage, key) for stage, keys in share_over.items() for key in keys}
    groups: Dict[Tuple, List[int]] = {}
    is_reference: List[bool] = []
    for i, labels in enumerate(model_labels):
        by_stage = split_labels_by_stage(labels, stage_sweeps)
        group_key = tuple(
            (stage, k, repr(v)) for stage, ls in by_stage.items() for k, v in ls if (stage, k) not in shared
        )
        groups.setdefault(group_key, []).append(i)
        is_reference.append(
            reference is not None
            and all(v == reference[stage][k] for stage, ls in by_stage.items() for k, v in ls if (stage, k) in shared)
        )

    ref_of = list(range(len(model_labels)))
    for group_key, members in groups.items():
        refs = [i for i in members if is_reference[i]] if reference is not None else members[:1]
        if len(refs) != 1:
            raise ValueError(
                f"{name} hp sharing: group {list(group_key)} has {len(refs)} models matching "
                f"{name}_hp_share_reference {dict(reference or {})}, expected exactly 1"
            )
        ref = refs[0]
        logger.info(
            f"{name} hp sharing: tuning once on {list(model_labels[ref])}, "
            f"reusing its HPs for {len(members) - 1} other model(s)"
        )
        for i in members:
            ref_of[i] = ref
    return ref_of
