from dataclasses import dataclass, asdict, field
from src.utils.data_roots import DataRoots
from prefect import flow
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
from typing import List, Dict, Any, Tuple
from itertools import product
from hydra.core.global_hydra import GlobalHydra
import os
import math
import numpy as np
import pandas as pd
import re
from src.prefect.caching import CacheableDict, CacheableDictConfig
from src.prefect.sweeping import get_hydra_sweep_combos
from src.prefect.task_options import task_options as remote_options
import logging
import sys
from typing import cast

from src.prefect.sweeping import (
    get_hydra_overwrite_list,
    get_hydra_sweep_combos,
    get_hydra_overwrite_str,
    get_pandas_dataframe_from_sweep_combos,
)

from src.tasks.plot_sweep_results import seaborn_plot_sweep_results_auto
from src.tasks.tuning_selection_task import tuning_selection_task

from prefect.logging import get_run_logger
from prefect import task
from prefect.futures import PrefectFuture

from src.utils.wandb_utils import (
    wandb_kwargs_for_prefect_task,
    gather_flow_infos_for_wandb,
    flatten_hydra_config,
)
from src.prefect.caching import hydra_config_to_cacheable_dict

from prefect import Task

@dataclass
class TuningConfig:
    task_config_path: str
    task_config_name: str
    tuning_config_path: str
    tuning_config_name: str
    tuning_overrides: List[str]
    tuning_cache_refresh: bool
    tuning_num_gpus: int
    task_overrides: List[str]
    task_sweep: Dict[str, List[Any] | Any]
    task_hp_overrides: List[str]
    task_hp_sweep_tune_basic_key: str
    task_hp_sweep_tune_extra: Dict[str, List[Any] | Any]
    task_hp_sweep_aggregate: Dict[str, List[Any] | Any]
    task_hp_log_wandb : bool
    task_run_overrides: List[str]
    task_run_sweep: Dict[str, List[Any] | Any]
    task_visual_run_overrides_extra: List[str]
    task_visual_run_sweep: Dict[str, List[Any] | Any]
    task_visual_cache_refresh: bool
    task_visual_num_gpus: int
    task_cache_refresh: bool
    task_common_post_overrides: List[str]
    task_num_gpus: int
    task_hp_budget_trials_per_param: int | None = None
    task_hp_refine_budget_trials_per_param: int | None = None
    # CPU counts per stage (Ray num_cpus); None -> Ray default.
    tuning_cpus_per_task: int | None = None
    task_cpus_per_task: int | None = None
    task_visual_cpus_per_task: int | None = None


def _is_bound_spec(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return "low" in value and "high" in value


def _is_numeric_list_sweep(value: Any) -> bool:
    if not isinstance(value, list) or len(value) < 2:
        return False
    for element in value:
        if isinstance(element, bool) or not isinstance(element, (int, float)):
            return False
    return True


def _discretize_numeric_list(
    values: List[Any],
    n_points: int,
    discretization_mode: str = "auto",
) -> List[Any]:
    numeric_values = [float(v) for v in values]
    low = min(numeric_values)
    high = max(numeric_values)
    if math.isclose(low, high):
        return [values[0]]

    if n_points <= 1:
        interpolated: List[float] = [low]
    else:
        positive = low > 0.0 and high > 0.0
        mode = str(discretization_mode).strip().lower()
        if mode == "log":
            use_log = positive
        elif mode == "linear":
            use_log = False
        else:
            use_log = positive and (high / low) > 100.0
        if use_log:
            log_low = math.log10(low)
            log_high = math.log10(high)
            interpolated = [
                math.pow(10.0, log_low + i * (log_high - log_low) / (n_points - 1))
                for i in range(n_points)
            ]
        else:
            interpolated = [
                low + i * (high - low) / (n_points - 1)
                for i in range(n_points)
            ]

    all_int = all(isinstance(v, int) and not isinstance(v, bool) for v in values)
    if all_int:
        rounded = sorted(set(int(round(v)) for v in interpolated))
        if len(rounded) == 0:
            return [int(round(low))]
        return rounded
    return interpolated


def _product_of_counts(counts: Dict[str, int]) -> int:
    product = 1
    for value in counts.values():
        product *= max(1, int(value))
    return product


def _allocate_counts_with_budget(
    counts: Dict[str, int],
    expandable_keys: List[str],
    budget: int,
) -> Dict[str, int]:
    if budget <= 0 or len(expandable_keys) == 0:
        return counts

    current_counts = dict(counts)
    current_product = _product_of_counts(current_counts)
    if current_product >= budget:
        return current_counts

    while True:
        best_key = None
        best_product = current_product
        for key in expandable_keys:
            next_product = (current_product // current_counts[key]) * (current_counts[key] + 1)
            if next_product <= budget and next_product > best_product:
                best_product = next_product
                best_key = key

        if best_key is None:
            break

        current_counts[best_key] += 1
        current_product = best_product

    return current_counts


def _discretize_bound_spec(
    spec: Dict[str, Any],
    n_points: int,
    discretization_mode: str = "auto",
) -> List[Any]:
    low = float(spec["low"])
    high = float(spec["high"])
    if high < low:
        low, high = high, low

    if math.isclose(low, high):
        n_points = 1

    if n_points <= 1:
        values: List[float] = [low]
    else:
        mode = str(discretization_mode).strip().lower()
        if mode == "log":
            scale = "log"
        elif mode == "linear":
            scale = "linear"
        else:
            scale = str(spec.get("scale", "linear")).strip().lower()
        if scale == "log":
            if low <= 0.0:
                values = [low + i * (high - low) / (n_points - 1) for i in range(n_points)]
            else:
                log_low = math.log10(low)
                log_high = math.log10(high)
                values = [
                    math.pow(10.0, log_low + i * (log_high - log_low) / (n_points - 1))
                    for i in range(n_points)
                ]
        else:
            values = [low + i * (high - low) / (n_points - 1) for i in range(n_points)]

    dtype = str(spec.get("dtype", "float")).strip().lower()
    if dtype in ["int", "integer"]:
        rounded = sorted(set(int(round(v)) for v in values))
        if len(rounded) == 0:
            rounded = [int(round(low))]
        return rounded
    return values


def _build_budgeted_tuning_hp_sweep(
    hp_sweep: Dict[str, Any],
    task_hp_budget_trials_per_param: int | None,
    discretization_mode: str = "auto",
) -> Dict[str, Any]:
    if task_hp_budget_trials_per_param is None or task_hp_budget_trials_per_param <= 0:
        return hp_sweep

    tunable_numeric_keys: List[str] = []
    fixed_combo_count = 1
    count_per_key: Dict[str, int] = {}
    expandable_keys: List[str] = []

    for key, value in hp_sweep.items():
        if _is_bound_spec(value):
            tunable_numeric_keys.append(key)
            spec = cast(Dict[str, Any], value)
            if "num" in spec:
                count_per_key[key] = max(2, int(spec["num"]))
            else:
                count_per_key[key] = 2
                expandable_keys.append(key)
            continue

        if _is_numeric_list_sweep(value):
            tunable_numeric_keys.append(key)
            unique_values = sorted(set(cast(List[Any], value)))
            count_per_key[key] = max(2, len(unique_values))
            expandable_keys.append(key)
            continue

        if isinstance(value, list):
            fixed_combo_count *= max(1, len(value))

    if len(tunable_numeric_keys) == 0:
        return hp_sweep

    # task_hp_budget_trials_per_param is treated as a per-parameter budget:
    # effective budget = budget_per_param * num_tunable_params, then divided by fixed combos.
    budget_per_param = max(1, int(task_hp_budget_trials_per_param))
    n_tunable = max(1, len(tunable_numeric_keys))
    budget_for_numeric = max(1, budget_per_param * n_tunable // max(1, fixed_combo_count))
    count_per_key = _allocate_counts_with_budget(count_per_key, expandable_keys, budget_for_numeric)

    expanded_hp_sweep = dict(hp_sweep)
    for key in tunable_numeric_keys:
        value = hp_sweep[key]
        n_points = max(2, count_per_key[key])
        if _is_bound_spec(value):
            expanded_hp_sweep[key] = _discretize_bound_spec(
                cast(Dict[str, Any], value),
                n_points,
                discretization_mode=discretization_mode,
            )
        else:
            expanded_hp_sweep[key] = _discretize_numeric_list(
                cast(List[Any], value),
                n_points,
                discretization_mode=discretization_mode,
            )

    return expanded_hp_sweep


def _parse_best_overrides(overrides: List[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        if str(value).strip().lower() in ["true", "false"]:
            parsed[key] = str(value).strip().lower() == "true"
            continue
        try:
            number = float(value)
            if number.is_integer() and str(value).strip().isdigit():
                parsed[key] = int(number)
            else:
                parsed[key] = number
        except ValueError:
            parsed[key] = value
    return parsed


def _build_refined_tuning_hp_sweep(
    hp_sweep_base: Dict[str, Any],
    phase1_hp_sweep: Dict[str, Any],
    phase1_best_overrides: List[str],
    task_hp_refine_budget_trials_per_param: int | None,
    phase1_anchor_values: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if task_hp_refine_budget_trials_per_param is None or task_hp_refine_budget_trials_per_param <= 0:
        return phase1_hp_sweep

    tunable_keys = [
        key
        for key, value in phase1_hp_sweep.items()
        if _is_numeric_list_sweep(value)
    ]

    if len(tunable_keys) == 0:
        logging.info("Skipping phase-2 refinement because no tunable numeric parameters were found.")
        return phase1_hp_sweep

    if len(tunable_keys) == 1:
        logging.info(
            "Applying 1D phase-2 refinement for parameter '%s' based on the local phase-1 neighborhood.",
            tunable_keys[0],
        )

    best_values = _parse_best_overrides(phase1_best_overrides)
    if phase1_anchor_values is not None:
        best_values = dict(best_values) | dict(phase1_anchor_values)
    refined_sweep: Dict[str, Any] = dict(hp_sweep_base)
    refined_range_modes: Dict[str, str] = {}

    for key, value in phase1_hp_sweep.items():
        if not _is_numeric_list_sweep(value):
            continue

        sorted_vals = sorted(set(cast(List[Any], value)))
        if len(sorted_vals) < 2:
            continue

        best_value = best_values.get(key, sorted_vals[len(sorted_vals) // 2])
        try:
            best_value_float = float(best_value)
        except (TypeError, ValueError):
            continue

        distances = [abs(float(v) - best_value_float) for v in sorted_vals]
        nearest_idx = int(np.argmin(distances))

        if len(sorted_vals) >= 3:
            if nearest_idx == 0:
                neighborhood_indices = [0, 1, 2]
                logging.warning(
                    "Phase-1 best for '%s' lies on lower boundary (%s). Using neighborhood %s for refinement.",
                    key,
                    sorted_vals[nearest_idx],
                    [sorted_vals[i] for i in neighborhood_indices],
                )
            elif nearest_idx == len(sorted_vals) - 1:
                neighborhood_indices = [len(sorted_vals) - 3, len(sorted_vals) - 2, len(sorted_vals) - 1]
                logging.warning(
                    "Phase-1 best for '%s' lies on upper boundary (%s). Using neighborhood %s for refinement.",
                    key,
                    sorted_vals[nearest_idx],
                    [sorted_vals[i] for i in neighborhood_indices],
                )
            else:
                neighborhood_indices = [nearest_idx - 1, nearest_idx, nearest_idx + 1]
            left_idx = min(neighborhood_indices)
            right_idx = max(neighborhood_indices)
        else:
            left_idx = max(0, nearest_idx - 1)
            right_idx = min(len(sorted_vals) - 1, nearest_idx + 1)

        if left_idx == right_idx:
            continue

        low = float(sorted_vals[left_idx])
        high = float(sorted_vals[right_idx])
        if math.isclose(low, high):
            continue

        refinement_mode = "linear"
        if low > 0.0 and high > 0.0:
            ratio = high / low
            if ratio > 10.0:
                refinement_mode = "log"

        all_int = all(isinstance(v, int) and not isinstance(v, bool) for v in sorted_vals)
        refined_sweep[key] = {
            "low": low,
            "high": high,
            "dtype": "int" if all_int else "float",
            "scale": refinement_mode,
        }
        refined_range_modes[key] = refinement_mode

        logging.info(
            "Phase-2 refinement for '%s': neighborhood=[%s, %s], scale=%s",
            key,
            low,
            high,
            refinement_mode,
        )

    return _build_budgeted_tuning_hp_sweep(
        refined_sweep,
        task_hp_refine_budget_trials_per_param,
        discretization_mode="auto",
    )


def _result_to_dict_for_refinement(result: Any) -> Dict[str, Any]:
    if isinstance(result, tuple) and len(result) == 2:
        return _result_to_dict_for_refinement(result[0])

    if isinstance(result, CacheableDict):
        return dict(result.cfg)

    if isinstance(result, dict):
        return dict(result)

    cfg_value = getattr(result, "cfg", None)
    if cfg_value is not None:
        if isinstance(cfg_value, dict):
            return dict(cfg_value)
        container = OmegaConf.to_container(cfg_value, resolve=True)
        if isinstance(container, dict):
            return cast(Dict[str, Any], dict(container))

    return {}


def _get_phase1_anchor_values_from_results(
    tuning_result: List[Tuple[PrefectFuture, List[Tuple[str, Any]]]],
    tuning_cfg: DictConfig,
) -> Dict[str, Any]:
    if len(tuning_result) == 0:
        return {}

    resolved_results = [_result_to_dict_for_refinement(task_future.result()) for task_future, _ in tuning_result]
    if len(resolved_results) == 0:
        return {}

    df_results = pd.DataFrame(resolved_results)
    if df_results.empty:
        return {}

    numeric_metrics = df_results.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_metrics) == 0:
        return {}

    tune_metric_pattern = str(getattr(tuning_cfg, "tune_metric", "")).strip()
    if len(tune_metric_pattern) == 0:
        return {}

    pattern = re.compile(tune_metric_pattern)
    matching_metrics = [metric for metric in numeric_metrics if pattern.search(metric)]
    if len(matching_metrics) != 1:
        return {}

    tuning_metric = matching_metrics[0]
    tune_metric_take_max = bool(getattr(tuning_cfg, "tune_metric_take_max", True))

    secondary_mode = str(getattr(tuning_cfg, "secondary_mode", "none")).strip().lower()
    if secondary_mode not in ["none", "hard", "soft"]:
        secondary_mode = "none"

    secondary_metric_name: str | None = None
    secondary_metric_pattern = str(getattr(tuning_cfg, "secondary_metric", "")).strip()
    if secondary_mode != "none" and len(secondary_metric_pattern) > 0:
        secondary_pattern = re.compile(secondary_metric_pattern)
        secondary_matches = [metric for metric in numeric_metrics if secondary_pattern.search(metric)]
        if len(secondary_matches) >= 1:
            secondary_metric_name = secondary_matches[0]
        else:
            secondary_mode = "none"

    agg_metrics = [tuning_metric]
    if secondary_metric_name is not None and secondary_metric_name not in agg_metrics:
        agg_metrics.append(secondary_metric_name)

    df_sweep = get_pandas_dataframe_from_sweep_combos([combo for _, combo in tuning_result])
    sweep_tune_params = [col for col in df_sweep.columns.tolist() if col.startswith("tun:")]
    if len(sweep_tune_params) == 0:
        return {}

    df_plot = pd.concat([df_results[agg_metrics], df_sweep], axis=1)
    agg = df_plot.groupby(sweep_tune_params, dropna=False)[agg_metrics].agg(["mean"])
    agg.columns = [f"{metric}__{stat}" for metric, stat in agg.columns]
    df_plot_stats = agg.reset_index()

    metric_col = f"{tuning_metric}__mean"
    if metric_col not in df_plot_stats.columns or df_plot_stats.empty:
        return {}

    candidate_df = df_plot_stats.copy()
    selection_metric_col = metric_col

    if secondary_mode != "none" and secondary_metric_name is not None:
        secondary_metric_col = f"{secondary_metric_name}__mean"
        if secondary_metric_col in candidate_df.columns:
            secondary_vals = candidate_df[secondary_metric_col].astype(float)
            secondary_take_max = bool(getattr(tuning_cfg, "secondary_metric_take_max", False))
            secondary_max_dev = float(max(0.0, getattr(tuning_cfg, "secondary_metric_max_deviation", 0.0)))
            secondary_soft_lambda = float(max(0.0, getattr(tuning_cfg, "secondary_soft_penalty_lambda", 1.0)))

            if secondary_take_max:
                best_secondary_value = float(secondary_vals.max())
                degradation = np.maximum(0.0, best_secondary_value - secondary_vals)
            else:
                best_secondary_value = float(secondary_vals.min())
                degradation = np.maximum(0.0, secondary_vals - best_secondary_value)

            excess_degradation = np.maximum(0.0, degradation - secondary_max_dev)

            if secondary_mode == "hard":
                feasible_mask = excess_degradation <= 0.0
                feasible_df = candidate_df.loc[feasible_mask].reset_index(drop=True)
                if not feasible_df.empty:
                    candidate_df = feasible_df
                else:
                    logging.warning(
                        "Phase-1 refinement anchor fallback: hard secondary constraint removed all candidates. "
                        "Using unconstrained primary metric anchor instead."
                    )
            elif secondary_mode == "soft":
                if tune_metric_take_max:
                    candidate_df["__selection_objective"] = (
                        candidate_df[metric_col].astype(float) - secondary_soft_lambda * excess_degradation
                    )
                else:
                    candidate_df["__selection_objective"] = (
                        candidate_df[metric_col].astype(float) + secondary_soft_lambda * excess_degradation
                    )
                selection_metric_col = "__selection_objective"

    if candidate_df.empty or selection_metric_col not in candidate_df.columns:
        return {}

    best_idx = candidate_df[selection_metric_col].idxmax() if tune_metric_take_max else candidate_df[selection_metric_col].idxmin()
    best_row = candidate_df.loc[best_idx]

    return {
        param.replace("tun:", ""): best_row[param]
        for param in sweep_tune_params
        if param in best_row
    }

    
def tuning_flow_before_plot(
    logger,
    wandb_params,
    task_config_path: str,
    task_config_name: str,
    tuning_config_path: str,
    tuning_config_name: str,
    tuning_overrides: List[str],
    tuning_cache_refresh: bool,
    tuning_num_gpus: int,
    task_overrides: List[str],
    task_sweep: Dict[str, List[Any] | Any],
    task_hp_overrides: List[str],
    task_hp_sweep_tune_basic_key: str,
    task_hp_sweep_tune_extra: Dict[str, List[Any] | Any],
    task_hp_sweep_aggregate: Dict[str, List[Any] | Any],
    task_hp_log_wandb : bool,
    task_run_overrides: List[str],
    task_run_sweep: Dict[str, List[Any] | Any],
    task_visual_run_overrides_extra: List[str],
    task_visual_run_sweep: Dict[str, List[Any] | Any],
    task_visual_cache_refresh: bool,
    task_visual_num_gpus: int,
    task_cache_refresh: bool,
    task_common_post_overrides: List[str],
    task_num_gpus: int,
    task_hp_budget_trials_per_param: int | None,
    task_hp_refine_budget_trials_per_param: int | None,
    task_to_tune : Task,
    task_to_tunes_inputs : Dict[str, Any],
    wandb_cfg: DictConfig,
    data_roots: DataRoots,
    extra_combo_labels: List[Tuple[str, Any]] = [],
    tuning_cpus_per_task: int | None = None,
    task_cpus_per_task: int | None = None,
    task_visual_cpus_per_task: int | None = None,
    # hp sharing: shared_hp_choices skips tuning and reuses another model's choices; hp_choices_out receives ours.
    shared_hp_choices: List[Tuple[PrefectFuture, List[Tuple[str, Any]]]] | None = None,
    hp_choices_out: List[Tuple[PrefectFuture, List[Tuple[str, Any]]]] | None = None,
    ):

    logger.info(
        "Starting recon sweep flow with base_config_path: %s, base_config_name: %s",
        task_config_path,
        task_config_name,
    )

    if shared_hp_choices is None:
        # 1) tuning sweeps
        tuning_task_result_list = []  # list of dicts
        GlobalHydra.instance().clear()
        with initialize(
            config_path=os.path.join("../../../hydra", task_config_path), version_base="1.2"
        ):
            # perform recon for all sweep combinations
            for sweep_combo in get_hydra_sweep_combos(task_sweep):
                # the basic setup (e.g. TV regularization)

                print(f"Starting recon sweep combo: {sweep_combo}")

                tuning_task_result = (
                    []
                )  # results of the tuning tasks for this specific recon override (e.g. exp)
                initial_cfg = compose(
                    config_name=task_config_name,
                    overrides=task_overrides
                    + task_hp_overrides
                    + get_hydra_overwrite_list(sweep_combo)
                    + task_common_post_overrides,
                )

                if task_hp_sweep_tune_basic_key in initial_cfg:
                    task_hp_sweep_tune_basic = cast(Dict[str, List[Any]], OmegaConf.to_container(initial_cfg[
                        task_hp_sweep_tune_basic_key
                    ],resolve=True))
                else:
                    task_hp_sweep_tune_basic: Dict[str, List[Any]] = {}
                print(f" - For recon sweep combo: {sweep_combo}, basic tuning hp sweep is: {task_hp_sweep_tune_basic}, and extra tuning hp sweep is: {task_hp_sweep_tune_extra}")
                hp_sweep = task_hp_sweep_tune_basic | cast(Dict[str, Any], OmegaConf.to_container(task_hp_sweep_tune_extra, resolve=True))
                hp_sweep_phase1 = _build_budgeted_tuning_hp_sweep(
                    hp_sweep,
                    task_hp_budget_trials_per_param,
                    discretization_mode="log",
                )

                print(
                    f" - For recon sweep combo: {sweep_combo}, phase-1 tuning hp sweep is: {hp_sweep_phase1}"
                )
            
                # Count how many tuning combinations will be explored
                phase1_combo_count = 1
                for key, value in hp_sweep_phase1.items():
                    if isinstance(value, list):
                        phase1_combo_count *= len(value)
                aggregate_combo_count = 1
                for key, value in task_hp_sweep_aggregate.items():
                    if isinstance(value, list):
                        aggregate_combo_count *= len(value)
                total_phase1_trials = phase1_combo_count * aggregate_combo_count
            
                logging.info(
                    "Phase-1 HP tuning structure: %d HP combos × %d aggregate combos = %d total trials. "
                    "Tuning params: %s",
                    phase1_combo_count,
                    aggregate_combo_count,
                    total_phase1_trials,
                    {k: (len(v) if isinstance(v, list) else 1) for k, v in hp_sweep_phase1.items()},
                )

                for sweep_combo_tuning in get_hydra_sweep_combos(hp_sweep_phase1):
                # e.g. some values for TV hyperparameter

                    print(
                        f" - - Starting recon hp tuning sweep combo: {sweep_combo_tuning}"
                    )

                    for sweep_combo_tuning_aggregate in get_hydra_sweep_combos(task_hp_sweep_aggregate):
                        # usually over sample indices

                        print(
                            f" - - - Starting recon hp tuning aggregate sweep combo: {sweep_combo_tuning_aggregate}"
                        )

                        cfg = compose(
                            config_name=task_config_name,
                            overrides=task_overrides
                            + task_hp_overrides
                            + get_hydra_overwrite_list(sweep_combo)
                            + get_hydra_overwrite_list(sweep_combo_tuning)
                            + get_hydra_overwrite_list(sweep_combo_tuning_aggregate)
                            + task_common_post_overrides,
                        )

                        wandb_params_task = wandb_params.model_copy(
                            update={
                                "name_aux": get_hydra_overwrite_str(sweep_combo)
                                + "_"
                                + get_hydra_overwrite_str(sweep_combo_tuning)
                                + "_"
                                + get_hydra_overwrite_str(sweep_combo_tuning_aggregate),
                                "config": flatten_hydra_config(cfg),
                                **cast(Dict[str, Any], wandb_cfg),
                            }
                        ).model_copy(update={"log": bool(task_hp_log_wandb)})

                        with remote_options(num_gpus=task_num_gpus, cpus_per_task=task_cpus_per_task):
                            logging.basicConfig(stream=sys.stdout, level=logging.INFO)
                            tuning_task_result.append(
                                (
                                    task_to_tune.with_options(
                                        refresh_cache=task_cache_refresh
                                    ).submit(
                                        **hydra_config_to_cacheable_dict(cfg),
                                        data_roots=data_roots,
                                        wandb_params_task=wandb_params_task,
                                        **task_to_tunes_inputs
                                    ),
                                    [(f"tun:{x}",y) for x,y in sweep_combo_tuning] + [(f"agg:{x}",y) for x,y in sweep_combo_tuning_aggregate]
                                )
                            )

                tuning_task_result_list.append(
                    {
                        "tuning_result": tuning_task_result,
                        "sweep_combo": sweep_combo,
                        "hp_sweep_base": hp_sweep,
                        "hp_sweep_phase1": hp_sweep_phase1,
                    }
                )

        # 2) phase-1 tuning aggregations
        tuning_hp_choice_overrides_stage1 = []
        tuning_cfg_for_refinement: DictConfig | None = None
        GlobalHydra.instance().clear()
        with initialize(
            config_path=os.path.join("../../../hydra", tuning_config_path), version_base="1.2"
        ):
            tuning_cfg = compose(config_name=tuning_config_name, overrides=tuning_overrides)
            tuning_cfg_for_refinement = tuning_cfg

            for tuning_stage in tuning_task_result_list:

                tuning_result = cast(List[Tuple[PrefectFuture, List[Tuple[str, Any]]]], tuning_stage["tuning_result"])
                sweep_combo = cast(List[Tuple[str, Any]], tuning_stage["sweep_combo"])
                hp_sweep_base = cast(Dict[str, Any], tuning_stage["hp_sweep_base"])
                hp_sweep_phase1 = cast(Dict[str, Any], tuning_stage["hp_sweep_phase1"])

                tuning_result_futures = [result[0] for result in tuning_result]
                tuning_df_sweep_combos = get_pandas_dataframe_from_sweep_combos(
                    [result[1] for result in tuning_result]
                )

                wandb_params_task = wandb_params.model_copy(
                    update={
                        "name_aux": get_hydra_overwrite_str(sweep_combo),
                        "config": flatten_hydra_config(tuning_cfg),
                        **cast(Dict[str, Any], wandb_cfg),
                    }
                )

                with remote_options(num_gpus=tuning_num_gpus, cpus_per_task=tuning_cpus_per_task):
                    tuning_hp_choice_overrides_stage1.append(
                        (
                            tuning_selection_task.with_options(
                                refresh_cache=tuning_cache_refresh
                            ).submit(
                                results=tuning_result_futures,
                                df_sweep=tuning_df_sweep_combos,
                                wandb_params_task=wandb_params_task,
                                tuning_phase_label="phase1",
                                **hydra_config_to_cacheable_dict(tuning_cfg),
                            ),
                            sweep_combo,
                            hp_sweep_base,
                            hp_sweep_phase1,
                            tuning_result,
                        )
                    )

        # 2b) optional adaptive refinement stage around phase-1 best
        tuning_task_result_list_phase2 = []
        if task_hp_refine_budget_trials_per_param is not None and task_hp_refine_budget_trials_per_param > 0:
            GlobalHydra.instance().clear()
            with initialize(
                config_path=os.path.join("../../../hydra", task_config_path), version_base="1.2"
            ):
                for (
                    tuning_hp_choice_overrides_phase1,
                    sweep_combo,
                    hp_sweep_base,
                    hp_sweep_phase1,
                    phase1_tuning_result,
                ) in tuning_hp_choice_overrides_stage1:
                    phase1_overrides = tuning_hp_choice_overrides_phase1.result().cfg
                    phase1_anchor_values = {}
                    if tuning_cfg_for_refinement is not None:
                        phase1_anchor_values = _get_phase1_anchor_values_from_results(
                            tuning_result=cast(List[Tuple[PrefectFuture, List[Tuple[str, Any]]]], phase1_tuning_result),
                            tuning_cfg=tuning_cfg_for_refinement,
                        )
                    hp_sweep_phase2 = _build_refined_tuning_hp_sweep(
                        hp_sweep_base=cast(Dict[str, Any], hp_sweep_base),
                        phase1_hp_sweep=cast(Dict[str, Any], hp_sweep_phase1),
                        phase1_best_overrides=cast(List[str], phase1_overrides),
                        task_hp_refine_budget_trials_per_param=task_hp_refine_budget_trials_per_param,
                        phase1_anchor_values=phase1_anchor_values,
                    )

                    print(
                        f" - For recon sweep combo: {sweep_combo}, phase-2 refinement hp sweep is: {hp_sweep_phase2}"
                    )

                    tuning_task_result_phase2 = []
                    for sweep_combo_tuning in get_hydra_sweep_combos(hp_sweep_phase2):
                        print(
                            f" - - Starting phase-2 recon hp tuning sweep combo: {sweep_combo_tuning}"
                        )
                        for sweep_combo_tuning_aggregate in get_hydra_sweep_combos(task_hp_sweep_aggregate):
                            cfg_phase2 = compose(
                                config_name=task_config_name,
                                overrides=task_overrides
                                + task_hp_overrides
                                + get_hydra_overwrite_list(cast(List[Tuple[str, Any]], sweep_combo))
                                + get_hydra_overwrite_list(sweep_combo_tuning)
                                + get_hydra_overwrite_list(sweep_combo_tuning_aggregate)
                                + task_common_post_overrides,
                            )

                            wandb_params_task_phase2 = wandb_params.model_copy(
                                update={
                                    "name_aux": get_hydra_overwrite_str(cast(List[Tuple[str, Any]], sweep_combo))
                                    + "_phase2_"
                                    + get_hydra_overwrite_str(sweep_combo_tuning)
                                    + "_"
                                    + get_hydra_overwrite_str(sweep_combo_tuning_aggregate),
                                    "config": flatten_hydra_config(cfg_phase2),
                                    **cast(Dict[str, Any], wandb_cfg),
                                }
                            ).model_copy(update={"log": bool(task_hp_log_wandb)})

                            with remote_options(num_gpus=task_num_gpus, cpus_per_task=task_cpus_per_task):
                                logging.basicConfig(stream=sys.stdout, level=logging.INFO)
                                tuning_task_result_phase2.append(
                                    (
                                        task_to_tune.with_options(
                                            refresh_cache=task_cache_refresh
                                        ).submit(
                                            **hydra_config_to_cacheable_dict(cfg_phase2),
                                            data_roots=data_roots,
                                            wandb_params_task=wandb_params_task_phase2,
                                            **task_to_tunes_inputs
                                        ),
                                        [(f"tun:{x}", y) for x, y in sweep_combo_tuning] + [(f"agg:{x}", y) for x, y in sweep_combo_tuning_aggregate],
                                    )
                                )

                    tuning_task_result_list_phase2.append((tuning_task_result_phase2, sweep_combo, phase1_tuning_result, phase1_overrides))

        # 2c) final tuning selection source (phase-2 if enabled, else phase-1)
        if task_hp_refine_budget_trials_per_param is not None and task_hp_refine_budget_trials_per_param > 0:
            tuning_hp_choice_overrides_list = []
            GlobalHydra.instance().clear()
            with initialize(
                config_path=os.path.join("../../../hydra", tuning_config_path), version_base="1.2"
            ):
                tuning_cfg = compose(config_name=tuning_config_name, overrides=tuning_overrides)
                for tuning_result, sweep_combo, phase1_tuning_result_ref, phase1_overrides_ref in tuning_task_result_list_phase2:
                    tuning_result_futures = [result[0] for result in tuning_result]
                    tuning_df_sweep_combos = get_pandas_dataframe_from_sweep_combos(
                        [result[1] for result in tuning_result]
                    )
                    phase1_result_futures_for_ref = [r[0] for r in phase1_tuning_result_ref]
                    phase1_df_for_ref = get_pandas_dataframe_from_sweep_combos(
                        [r[1] for r in phase1_tuning_result_ref]
                    )

                    wandb_params_task = wandb_params.model_copy(
                        update={
                            "name_aux": get_hydra_overwrite_str(cast(List[Tuple[str, Any]], sweep_combo)),
                            "config": flatten_hydra_config(tuning_cfg),
                            **cast(Dict[str, Any], wandb_cfg),
                        }
                    )

                    with remote_options(num_gpus=tuning_num_gpus, cpus_per_task=tuning_cpus_per_task):
                        tuning_hp_choice_overrides_list.append(
                            (
                                tuning_selection_task.with_options(
                                    refresh_cache=tuning_cache_refresh
                                ).submit(
                                    results=tuning_result_futures,
                                    df_sweep=tuning_df_sweep_combos,
                                    wandb_params_task=wandb_params_task,
                                    tuning_phase_label="phase2",
                                    phase1_reference_results=phase1_result_futures_for_ref,
                                    phase1_reference_df_sweep=phase1_df_for_ref,
                                    phase1_reference_best_overrides=phase1_overrides_ref,
                                    **hydra_config_to_cacheable_dict(tuning_cfg),
                                ),
                                sweep_combo,
                            )
                        )
        else:
            tuning_hp_choice_overrides_list = [
                (tuning_hp_choice_overrides, sweep_combo)
                # stage-1 entries: (selection_task, sweep_combo, hp_sweep_base, hp_sweep_phase1, tuning_result)
                for tuning_hp_choice_overrides, sweep_combo, _, _, _ in tuning_hp_choice_overrides_stage1
            ]
    else:
        # hp sharing: reuse the reference model's selected HPs (one entry per task_sweep combo, same order)
        expected_combos = get_hydra_sweep_combos(task_sweep)
        assert [c for _, c in shared_hp_choices] == expected_combos, (
            f"shared_hp_choices combos {[c for _, c in shared_hp_choices]} do not match task_sweep combos {expected_combos}"
        )
        tuning_hp_choice_overrides_list = list(shared_hp_choices)
        GlobalHydra.instance().clear()
    if hp_choices_out is not None:
        hp_choices_out.extend(tuning_hp_choice_overrides_list)

    # 3) final recon runs with tuned hyperparameters
    task_results = []
    visual_results = []
    with initialize(
        config_path=os.path.join("../../../hydra", task_config_path), version_base="1.2"
    ):
        for tuning_hp_choice_overrides, sweep_combo in tuning_hp_choice_overrides_list:

            # here we need to wait for the results, so that we can resolve them
            tuning_hp_choice_overrides = tuning_hp_choice_overrides.result().cfg

            print(
                f"Starting final recon runs for recon sweep combo: {sweep_combo} with tuning hp choices: {tuning_hp_choice_overrides}"
            )

            for sweep_combo_run in get_hydra_sweep_combos(task_run_sweep):

                print(
                    f"Starting recon run sweep combo: {sweep_combo_run} for recon sweep combo: {sweep_combo} and tuning hp choice: {tuning_hp_choice_overrides}"
                )

                cfg = compose(
                    config_name=task_config_name,
                    overrides=task_overrides
                    + task_run_overrides
                    + get_hydra_overwrite_list(sweep_combo)
                    + get_hydra_overwrite_list(sweep_combo_run)
                    + task_common_post_overrides
                    + tuning_hp_choice_overrides
                )

                wandb_params_task = wandb_params.model_copy(
                    update={
                        "name_aux": get_hydra_overwrite_str(sweep_combo)
                        + "_"
                        + get_hydra_overwrite_str(sweep_combo_run),
                        "config": flatten_hydra_config(cfg),
                        **cast(Dict[str, Any], wandb_cfg),
                    }
                )

                with remote_options(num_gpus=task_num_gpus, cpus_per_task=task_cpus_per_task):
                    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
                    task_results.append(
                        (
                            task_to_tune.with_options(
                                refresh_cache=task_cache_refresh
                            ).submit(
                                **hydra_config_to_cacheable_dict(cfg),
                                data_roots=data_roots,
                                wandb_params_task=wandb_params_task,
                                **task_to_tunes_inputs
                            ),
                            extra_combo_labels + sweep_combo + sweep_combo_run,
                        )
                    )

            for sweep_combo_visual in get_hydra_sweep_combos(task_visual_run_sweep):

                print(
                    f"Starting visual run sweep combo: {sweep_combo_visual} for recon sweep combo: {sweep_combo} and tuning hp choice: {tuning_hp_choice_overrides}"
                )

                visual_cfg = compose(
                    config_name=task_config_name,
                    overrides=task_overrides
                    + task_run_overrides
                    + task_visual_run_overrides_extra
                    + get_hydra_overwrite_list(sweep_combo)
                    + get_hydra_overwrite_list(sweep_combo_visual)
                    + task_common_post_overrides
                    + tuning_hp_choice_overrides,
                )

                visual_wandb_params_task = wandb_params.model_copy(
                    update={
                        "name_aux": get_hydra_overwrite_str(sweep_combo)
                        + "_visual_"
                        + get_hydra_overwrite_str(sweep_combo_visual),
                        "config": flatten_hydra_config(visual_cfg),
                        **cast(Dict[str, Any], wandb_cfg),
                    }
                )

                with remote_options(num_gpus=task_visual_num_gpus, cpus_per_task=task_visual_cpus_per_task):
                    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
                    visual_results.append(
                        (
                            task_to_tune.with_options(
                                refresh_cache=task_visual_cache_refresh
                            ).submit(
                                **hydra_config_to_cacheable_dict(visual_cfg),
                                data_roots=data_roots,
                                wandb_params_task=visual_wandb_params_task,
                                **task_to_tunes_inputs
                            ),
                            extra_combo_labels + sweep_combo + sweep_combo_visual,
                        )
                    )

    return task_results, visual_results

from src.utils.wandb_utils import WandbParamsTask
from src.tasks.recon_tasks import recon_task
from src.tasks.recon_baseline_task import recon_baseline_task

def tune_recon_and_baselines(
    recon_recon_config_path: str,
    recon_recon_config_name: str,
    recon_tuning_config_path: str,
    recon_tuning_config_name: str,
    recon_tuning_overrides: List[str],
    recon_tuning_cache_refresh: bool,
    recon_tuning_num_gpus: int,
    common_hp_sweep_aggregate: Dict[str, List[Any] | Any],
    common_run_sweep: Dict[str, List[Any] | Any],
    common_recon_overrides: List[str],
    common_post_overrides: List[str],
    recon_recon_sweep: Dict[str, List[Any] | Any],
    recon_recon_hp_overrides: List[str],
    recon_recon_hp_sweep_tune_basic_key: str,
    recon_recon_hp_sweep_tune_extra: Dict[str, List[Any] | Any],
    recon_recon_hp_log_wandb : bool,
    recon_recon_run_overrides: List[str],
    recon_recon_cache_refresh: bool,
    recon_recon_num_gpus: int,
    baseline_recon_config_path: str,
    baseline_recon_config_name: str,
    baseline_tuning_config_path: str,
    baseline_tuning_config_name: str,
    baseline_tuning_overrides: List[str],
    baseline_tuning_cache_refresh: bool,
    baseline_tuning_num_gpus: int,
    baseline_recon_sweep: Dict[str, List[Any] | Any],
    baseline_recon_hp_overrides: List[str],
    baseline_recon_hp_sweep_tune_basic_key: str,
    baseline_recon_hp_sweep_tune_extra: Dict[str, List[Any] | Any],
    baseline_recon_hp_log_wandb : bool,
    baseline_recon_run_overrides: List[str],
    baseline_recon_cache_refresh: bool,
    baseline_recon_num_gpus: int,
    wandb_cfg: DictConfig,
    data_roots: DataRoots,
    logger,
    wandb_params : WandbParamsTask,
    task_to_tune_recon_input : Dict[str, Any] = {},
    task_to_tune_baseline_input : Dict[str, Any] = {},
    skip_baseline : bool = False,
    skip_recon : bool = False, # run only the baselines
    # per-arm final run sweeps (default common_run_sweep), for keys that exist in only one config schema
    recon_run_sweep: Dict[str, List[Any] | Any] | None = None,
    baseline_run_sweep: Dict[str, List[Any] | Any] | None = None,
    recon_recon_hp_budget_trials_per_param: int | None = None,
    recon_recon_hp_refine_budget_trials_per_param: int | None = None,
    baseline_recon_hp_budget_trials_per_param: int | None = None,
    baseline_recon_hp_refine_budget_trials_per_param: int | None = None,
    common_visual_run_sweep: Dict[str, List[Any] | Any] | None = None,
    common_visual_recon_overrides_extra: List[str] | None = None,
    common_visual_recon_save_result_enable: bool = True,
    common_visual_recon_cache_refresh: bool = False,
    common_visual_recon_num_gpus: int = 1,
    # CPU counts per stage (Ray num_cpus); None -> Ray default.
    recon_tuning_cpus_per_task: int | None = None,
    recon_recon_cpus_per_task: int | None = None,
    baseline_tuning_cpus_per_task: int | None = None,
    baseline_recon_cpus_per_task: int | None = None,
    common_visual_recon_cpus_per_task: int | None = None,
    extra_combo_labels: List[Tuple[str, Any]] = [],
    # hp sharing for the recon arm, see tuning_flow_before_plot's shared_hp_choices/hp_choices_out
    recon_shared_hp_choices: List[Tuple[PrefectFuture, List[Tuple[str, Any]]]] | None = None,
    recon_hp_choices_out: List[Tuple[PrefectFuture, List[Tuple[str, Any]]]] | None = None,
) -> Tuple[
    List[Tuple[PrefectFuture, List[Tuple[str, Any]]]],
    List[Tuple[PrefectFuture, List[Tuple[str, Any]]]],
]:

    visual_save_override = f"++save_result.enable={str(common_visual_recon_save_result_enable)}"
    visual_run_overrides_extra = list(common_visual_recon_overrides_extra or []) + [visual_save_override]
    visual_run_sweep = common_visual_run_sweep or {}

    recon_tuning_config = TuningConfig(
        task_config_path=recon_recon_config_path,
        task_config_name=recon_recon_config_name,
        tuning_config_path=recon_tuning_config_path,
        tuning_config_name=recon_tuning_config_name,
        tuning_overrides=recon_tuning_overrides,
        tuning_cache_refresh=recon_tuning_cache_refresh,
        tuning_num_gpus=recon_tuning_num_gpus,
        task_overrides=common_recon_overrides,
        task_sweep=recon_recon_sweep,
        task_hp_overrides=recon_recon_hp_overrides,
        task_hp_sweep_tune_basic_key=recon_recon_hp_sweep_tune_basic_key,
        task_hp_sweep_tune_extra=recon_recon_hp_sweep_tune_extra,
        task_hp_sweep_aggregate=common_hp_sweep_aggregate,
        task_hp_log_wandb=recon_recon_hp_log_wandb,
        task_run_overrides=recon_recon_run_overrides,
        task_run_sweep=recon_run_sweep if recon_run_sweep is not None else common_run_sweep,
        task_visual_run_overrides_extra=visual_run_overrides_extra,
        task_visual_run_sweep=visual_run_sweep,
        task_visual_cache_refresh=common_visual_recon_cache_refresh,
        task_visual_num_gpus=common_visual_recon_num_gpus,
        task_cache_refresh=recon_recon_cache_refresh,
        task_common_post_overrides=common_post_overrides,
        task_num_gpus=recon_recon_num_gpus,
        task_hp_budget_trials_per_param=recon_recon_hp_budget_trials_per_param,
        task_hp_refine_budget_trials_per_param=recon_recon_hp_refine_budget_trials_per_param,
        tuning_cpus_per_task=recon_tuning_cpus_per_task,
        task_cpus_per_task=recon_recon_cpus_per_task,
        task_visual_cpus_per_task=common_visual_recon_cpus_per_task,
    )

    baseline_recon_tuning_config = TuningConfig(
        task_config_path=baseline_recon_config_path,
        task_config_name=baseline_recon_config_name,
        tuning_config_path=baseline_tuning_config_path,
        tuning_config_name=baseline_tuning_config_name,
        tuning_overrides=baseline_tuning_overrides,
        tuning_cache_refresh=baseline_tuning_cache_refresh,
        tuning_num_gpus=baseline_tuning_num_gpus,
        task_overrides=common_recon_overrides,
        task_sweep=baseline_recon_sweep,
        task_hp_overrides=baseline_recon_hp_overrides,
        task_hp_sweep_tune_basic_key=baseline_recon_hp_sweep_tune_basic_key,
        task_hp_sweep_tune_extra=baseline_recon_hp_sweep_tune_extra,
        task_hp_sweep_aggregate=common_hp_sweep_aggregate,
        task_hp_log_wandb=baseline_recon_hp_log_wandb,
        task_run_overrides=baseline_recon_run_overrides,
        task_run_sweep=baseline_run_sweep if baseline_run_sweep is not None else common_run_sweep,
        task_visual_run_overrides_extra=visual_run_overrides_extra,
        task_visual_run_sweep=visual_run_sweep,
        task_visual_cache_refresh=common_visual_recon_cache_refresh,
        task_visual_num_gpus=common_visual_recon_num_gpus,
        task_cache_refresh=baseline_recon_cache_refresh,
        task_common_post_overrides=common_post_overrides,
        task_num_gpus=baseline_recon_num_gpus,
        task_hp_budget_trials_per_param=baseline_recon_hp_budget_trials_per_param,
        task_hp_refine_budget_trials_per_param=baseline_recon_hp_refine_budget_trials_per_param,
        tuning_cpus_per_task=baseline_tuning_cpus_per_task,
        task_cpus_per_task=baseline_recon_cpus_per_task,
        task_visual_cpus_per_task=common_visual_recon_cpus_per_task,
    ) 

    if skip_recon and skip_baseline:
        raise ValueError("tune_recon_and_baselines: skip_recon and skip_baseline cannot both be True")
    task_to_tune_list = []
    task_to_tunes_inputs_list = []
    tuning_configs = []
    shared_hp_choices_list = []
    hp_choices_out_list = []
    if not skip_recon:
        task_to_tune_list.append(recon_task)
        task_to_tunes_inputs_list.append(task_to_tune_recon_input)
        tuning_configs.append(recon_tuning_config)
        shared_hp_choices_list.append(recon_shared_hp_choices)
        hp_choices_out_list.append(recon_hp_choices_out)
    if not skip_baseline:
        task_to_tune_list.append(recon_baseline_task)
        task_to_tunes_inputs_list.append(task_to_tune_baseline_input)
        tuning_configs.append(baseline_recon_tuning_config)
        shared_hp_choices_list.append(None)
        hp_choices_out_list.append(None)

    all_task_results, all_visual_results = tuning_flow(
        tuning_configs = tuning_configs,
        wandb_cfg=wandb_cfg,
        data_roots=data_roots,
        task_to_tune=task_to_tune_list,
        task_to_tunes_inputs=task_to_tunes_inputs_list,
        wandb_task_params=wandb_params,
        logger=logger,
        extra_combo_labels=extra_combo_labels,
        shared_hp_choices_list=shared_hp_choices_list,
        hp_choices_out_list=hp_choices_out_list,
    )

    return all_task_results, all_visual_results

def tuning_flow(
    tuning_configs : List[TuningConfig],
    wandb_cfg: DictConfig,
    data_roots: DataRoots,
    task_to_tune : List[Task],
    wandb_task_params : WandbParamsTask,
    task_to_tunes_inputs : List[Dict[str, Any]],
    logger,
    extra_combo_labels: List[Tuple[str, Any]] = [],
    # per tuning_config: shared_hp_choices / hp_choices_out of tuning_flow_before_plot (None = off)
    shared_hp_choices_list: List[Any] | None = None,
    hp_choices_out_list: List[Any] | None = None,
) -> Tuple[
    List[Tuple[PrefectFuture, List[Tuple[str, Any]]]],
    List[Tuple[PrefectFuture, List[Tuple[str, Any]]]],
]:

    assert len(task_to_tune) == len(tuning_configs), f"Length of task_to_tune ({len(task_to_tune)}) should be the same as number of tuning_configs ({len(tuning_configs)})"    

    all_task_results = []
    all_visual_results = []

    shared_hp_choices_list = shared_hp_choices_list or [None] * len(tuning_configs)
    hp_choices_out_list = hp_choices_out_list or [None] * len(tuning_configs)
    for tuning_config, task, task_inputs, shared_hp_choices, hp_choices_out in zip(
        tuning_configs, task_to_tune, task_to_tunes_inputs, shared_hp_choices_list, hp_choices_out_list
    ):
        task_results, visual_results = tuning_flow_before_plot(
            **asdict(tuning_config),
            task_to_tune=task,
            wandb_cfg=wandb_cfg,
            data_roots=data_roots,
            logger=logger,
            wandb_params=wandb_task_params,
            task_to_tunes_inputs=task_inputs,
            extra_combo_labels=extra_combo_labels,
            shared_hp_choices=shared_hp_choices,
            hp_choices_out=hp_choices_out,
        )
        all_task_results.extend(task_results)
        all_visual_results.extend(visual_results)

    return all_task_results, all_visual_results