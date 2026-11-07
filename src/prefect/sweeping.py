
from typing import Dict, List, Any, Tuple
from itertools import product
import hashlib
import re
from pandas import DataFrame

LINKED_KEY_SEPARATOR = "&"


def get_hydra_sweep_combos(overrides: Dict[str, List[Any]] | Dict[str, str]) -> List[List[Tuple[str, str]]]:
    """
        Cartesian products for overrides of the form:
        - +param1: [value1, value2]
        - +param2: [value3, value4]

        Output: [[(+param1, value1), (+param2, value3)], [(+param1, value1), (+param2, value4)], [(+param1, value2), (+param2, value3)], [(+param1, value2), (+param2, value4)]]

        Linked (zipped) parameters: several override keys can be joined into one sweep key
        with "&", in which case each of that key's values is a list holding one value per
        joined key, and those values move together instead of being crossed with each other:

        - "+param1&+param2": [[value1, value3], [value2, value4]]

        Output: [[(+param1, value1), (+param2, value3)], [(+param1, value2), (+param2, value4)]]
    """
    keys = list(overrides.keys())
    values = []
    for key in keys:
        if not isinstance(overrides[key], List):
            l = list(eval(str(overrides[key])))
            print(f"Parsing override {key} with value {overrides[key]} into list: {l}")
            values.append(
                l
                 # purpose is to parse list(range(0,5))
            ) # this can fail
        else:
            values.append(overrides[key])

    combos = []
    for combo in product(*values):
        pairs: List[Tuple[str, str]] = []
        for key, value in zip(keys, combo):
            if LINKED_KEY_SEPARATOR not in key:
                pairs.append((key, value))
                continue
            linked_keys = [k.strip() for k in key.split(LINKED_KEY_SEPARATOR) if k.strip()]
            linked_values = list(value) if isinstance(value, (list, tuple)) else list(eval(str(value)))
            if len(linked_keys) != len(linked_values):
                raise ValueError(
                    f"Linked sweep key {key!r} joins {len(linked_keys)} parameters but got "
                    f"{len(linked_values)} value(s) ({value!r}); each entry of a linked key's "
                    f"value list must hold exactly one value per joined parameter."
                )
            pairs.extend(zip(linked_keys, linked_values))
        combos.append(pairs)
    return combos

def get_hydra_overwrite_list(sweep_combo: List[Tuple[str, str]]) -> List[str]:
    return [f"{k}={v}" for k, v in sweep_combo]

def _sanitize_name_token(token: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "-", token)
    token = re.sub(r"-+", "-", token).strip("-_")
    return token or "x"


def _shorten_key_name(key: str) -> str:
    key = key.lstrip("+-")
    parts = [part for part in key.split(".") if part and part not in {"params", "cfg"}]
    if not parts:
        return "param"

    parts = parts[-3:]
    alias_map = {
        "initializer": "init",
        "propagation": "prop",
        "baseline_params": "base",
        "num_inference_steps": "steps",
        "propagation_mode": "mode",
        "sample_index": "sample",
        "domain_specific_setup": "domain",
    }
    aliased_parts = [alias_map.get(part, part) for part in parts]
    return _sanitize_name_token("_".join(aliased_parts))


def _stringify_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "-".join(_stringify_value(v) for v in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return _sanitize_name_token(str(value).lower())

def get_hydra_overwrite_str(sweep_combo: List[Tuple[str, str]]) -> str:
    name_parts = []
    for key, value in sweep_combo:
        short_key = _shorten_key_name(str(key))
        short_value = _stringify_value(value)
        name_parts.append(f"{short_key}-{short_value}")

    safe_name = "__".join(name_parts)
    if len(safe_name) <= 120:
        return safe_name

    name_hash = hashlib.md5(safe_name.encode("utf-8")).hexdigest()[:8]
    truncated = safe_name[:111].rstrip("-_")
    return f"{truncated}__{name_hash}"

def get_pandas_dataframe_from_sweep_combos(sweep_combos: List[List[Tuple[str, str]]]) -> DataFrame:
    """Converts a list of sweep combinations into a pandas DataFrame (missing keys become None)."""
    if len(sweep_combos) == 0:
        return DataFrame()

    column_names = []
    for combo in sweep_combos:
        for key, _ in combo:
            if key not in column_names:
                column_names.append(key)

    data = {key: [] for key in column_names}
    for combo in sweep_combos:
        combo_dict = dict(combo)
        for key in column_names:
            data[key].append(combo_dict.get(key, None))

    # Convert to DataFrame
    return DataFrame(data, columns=column_names)