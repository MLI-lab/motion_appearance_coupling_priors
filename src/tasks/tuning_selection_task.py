from prefect import flow, task
from prefect.cache_policies import TASK_SOURCE, INPUTS, NONE
import logging
import sys
import wandb
from src.utils.wandb_utils import wandb_kwargs_for_prefect_task, WandbParamsTask
from typing import List, Optional
from src.prefect.caching import CacheableDict, CacheableList, CacheableDictConfig, CacheableListConfig
from omegaconf import OmegaConf
import pandas as pd
import seaborn as sns
from typing import Tuple
import io
import math
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Dict, Any
from src.utils.data_roots import DataRoots


def _result_to_dict(result: Any) -> Dict[str, Any]:
    if isinstance(result, tuple) and len(result) == 2:
        metrics, storage = result
        result_dict = _result_to_dict(metrics)
        storage_path = None
        if hasattr(storage, "storage_path"):
            storage_path = getattr(storage, "storage_path")
        elif isinstance(storage, dict):
            storage_path = storage.get("storage_path")
        if storage_path is not None:
            result_dict["storage_path"] = storage_path
        return result_dict

    if isinstance(result, CacheableDict):
        return dict(result.cfg)

    if isinstance(result, CacheableDictConfig):
        container = OmegaConf.to_container(result.cfg, resolve=True)
        return container if isinstance(container, dict) else {"value": container}

    if isinstance(result, dict):
        return dict(result)

    cfg_value = getattr(result, "cfg", None)
    if cfg_value is not None:
        if isinstance(cfg_value, dict):
            return dict(cfg_value)
        if hasattr(cfg_value, "keys"):
            container = OmegaConf.to_container(cfg_value, resolve=True)
            return container if isinstance(container, dict) else {"value": container}
        return {"value": cfg_value}

    raise TypeError(f"Unsupported result type for tuning selection: {type(result)}")

@task(cache_policy=INPUTS - "data_roots", name="Tuning HP selection", tags=["tuning"], version="1.1", retries=0, description="Select the best hyperparameters based on tuning results.")
def tuning_selection_task(
        # df_results : pd.DataFrame,
        results : List[CacheableDict], # or rather PrefectFuture
        df_sweep : pd.DataFrame,
        tune_metric : str,
        tune_metric_take_max : bool,
        selection_method : str,
        add_wandb_plot : bool,
        log_lin_scale_auto_enabled: bool,
        log_lin_scale_threshold: float,
        apply_log_to_spline: bool,
        spline_smoothing_factor: float,
        spline_nr_points: int,
        fail_on_boundary_best: bool,
        dpi : int,
        wandb_params_task: WandbParamsTask,
        secondary_mode: str = "none",
        secondary_metric: str = "",
        secondary_metric_take_max: bool = False,
        secondary_metric_max_deviation: float = 0.0,
        secondary_soft_penalty_lambda: float = 1.0,
        tuning_phase_label: str = "",
        phase1_reference_results: Optional[List] = None,
        phase1_reference_df_sweep: Optional[pd.DataFrame] = None,
        phase1_reference_best_overrides: Optional[List[str]] = None,
    ) -> CacheableList: # CacheableDict[Dict[str, Any]] instead of CacheableDict[str, List[Any]]

    """

        Input:
            - results: List of CacheableDict, each containing the results of a tuning run, including the hyperparameters and the evaluation metric (e.g. validation loss).
            - df_sweep: DataFrame containing the sweep combinations and their corresponding hyperparameters.
            - tune_metric: The metric to use for selecting the best hyperparameters.
        Returns:
            - CacheableList: A list of selected hyperparameters, of the form ["++param.x.y.z=0.01","++param.a.b.c=0.1",...], just as it would be a override parameter in the hydra config.
    """

    ret : List[str] = []
    spline_plot_data: Dict[str, Any] | None = None

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    df_results = pd.DataFrame([_result_to_dict(result) for result in results])

    with wandb.init(**wandb_kwargs_for_prefect_task(wandb_params_task)):

        phase_label = str(tuning_phase_label).strip()

        def _phase_title(base_title: str) -> str:
            return f"{base_title} ({phase_label})" if len(phase_label) > 0 else base_title

        secondary_mode = str(secondary_mode).strip().lower()
        if secondary_mode not in ["none", "hard", "soft"]:
            logging.warning("Unsupported secondary_mode='%s'. Falling back to 'none'.", secondary_mode)
            secondary_mode = "none"

        secondary_metric_pattern = str(secondary_metric).strip()
        if secondary_mode != "none" and len(secondary_metric_pattern) == 0:
            logging.warning("secondary_mode is enabled but secondary_metric is empty. Falling back to 'none'.")
            secondary_mode = "none"

        secondary_max_dev = float(max(0.0, secondary_metric_max_deviation))
        secondary_soft_lambda = float(max(0.0, secondary_soft_penalty_lambda))

        # Identify numeric metrics in results
        numeric_metrics = df_results.select_dtypes(include=['number']).columns.tolist()
        print(f"Ignoring the following result params: {df_results.columns.difference(numeric_metrics).tolist()}")

        # find the tuning_metric from the numeric_metrics using regex, but make sure its unique otherwise throw an error listing the options
        import re
        pattern = re.compile(tune_metric)
        matching_metrics = [metric for metric in numeric_metrics if pattern.search(metric)]
        if len(matching_metrics) == 0:
            raise ValueError(f"No matching metric found for tuning_metric '{tune_metric}' in numeric metrics: {numeric_metrics}")
        elif len(matching_metrics) > 1:
            raise ValueError(f"Multiple matching metrics found for tuning_metric '{tune_metric}' in numeric metrics: {matching_metrics}. Please make sure the tuning_metric regex is specific enough to match only one metric.")
        else:
            tuning_metric = matching_metrics[0]
            print(f"Selected tuning metric: {tuning_metric}")

        secondary_tuning_metric = None
        if secondary_mode != "none":
            secondary_pattern = re.compile(secondary_metric_pattern)
            secondary_matching_metrics = [metric for metric in numeric_metrics if secondary_pattern.search(metric)]
            if len(secondary_matching_metrics) == 0:
                logging.warning(
                    "No matching metric found for secondary_metric '%s' in numeric metrics: %s. Falling back to 'none'.",
                    secondary_metric_pattern,
                    numeric_metrics,
                )
                secondary_mode = "none"
            else:
                if len(secondary_matching_metrics) > 1:
                    logging.warning(
                        "Multiple metrics found for secondary_metric '%s': %s. Using first match '%s'.",
                        secondary_metric_pattern,
                        secondary_matching_metrics,
                        secondary_matching_metrics[0],
                    )
                secondary_tuning_metric = secondary_matching_metrics[0]
                print(f"Selected secondary tuning metric: {secondary_tuning_metric}")

        agg_metrics = [tuning_metric]
        if secondary_tuning_metric is not None and secondary_tuning_metric not in agg_metrics:
            agg_metrics.append(secondary_tuning_metric)

        df_results_metric = df_results[agg_metrics]

        # Identify sweep parameters (numeric or categorical)
        sweep_params = df_sweep.columns.tolist()
        # we assume that sweep_param columns start with "tun:" or "agg:" to distinguish them from other params in df_results
        sweep_tune_params = [col for col in sweep_params if col.startswith("tun:")]
        sweep_aggregate_params = [col for col in sweep_params if col.startswith("agg:")]

        if len(sweep_tune_params) == 0:
            logging.warning(
                "No tuning HP parameters found in df_sweep (columns: %s). "
                "The HP sweep was empty — no hyperparameter overrides will be applied.",
                sweep_params,
            )
            return CacheableList(cfg=ret)

        df_plot = pd.concat([df_results_metric, df_sweep], axis=1)

        agg = df_plot.groupby(sweep_tune_params, dropna=False)[agg_metrics].agg(["mean", "std"])
        agg.columns = [f"{metric}__{stat}" for metric, stat in agg.columns]
        df_plot_stats = agg.reset_index()
        df_plot_stats_full = df_plot_stats.copy()
        
        # Log a summary of the explored HP combinations
        logging.info(
            "HP tuning aggregation complete: %d explored combinations of %s. "
            "Tuning metrics: %s. Shape: %s",
            len(df_plot_stats),
            sweep_tune_params,
            agg_metrics,
            df_plot_stats.shape,
        )
        if len(df_plot_stats) <= 3:
            logging.info("Detailed df_plot_stats:\n%s", df_plot_stats.to_string())
        
        metric_mean_cols = {metric: f"{metric}__mean" for metric in agg_metrics}
        metric_std_cols = {metric: f"{metric}__std" for metric in agg_metrics}

        primary_metric_col = metric_mean_cols[tuning_metric]
        selection_metric_col = primary_metric_col
        secondary_metric_col = None

        df_plot_stats_full["__secondary_feasible"] = True
        df_plot_stats_full["__secondary_excess_degradation"] = 0.0

        if secondary_mode != "none" and secondary_tuning_metric is not None:
            secondary_metric_col = metric_mean_cols[secondary_tuning_metric]
            secondary_vals_full = df_plot_stats_full[secondary_metric_col].astype(float)

            if secondary_metric_take_max:
                best_secondary_value = float(secondary_vals_full.max())
                degradation_full = np.maximum(0.0, best_secondary_value - secondary_vals_full)
            else:
                best_secondary_value = float(secondary_vals_full.min())
                degradation_full = np.maximum(0.0, secondary_vals_full - best_secondary_value)

            excess_degradation_full = np.maximum(0.0, degradation_full - secondary_max_dev)
            feasible_mask_full = excess_degradation_full <= 0.0
            df_plot_stats_full["__secondary_feasible"] = feasible_mask_full
            df_plot_stats_full["__secondary_excess_degradation"] = excess_degradation_full

            if secondary_mode == "hard":
                nr_feasible = int(feasible_mask_full.sum())
                if nr_feasible > 0:
                    df_plot_stats = df_plot_stats_full.loc[feasible_mask_full].reset_index(drop=True)
                    print(
                        f"Applied hard secondary constraint on '{secondary_tuning_metric}'. "
                        f"Keeping {nr_feasible}/{len(feasible_mask_full)} candidates with max_deviation={secondary_max_dev}."
                    )

                    if len(sweep_tune_params) == 1 and nr_feasible <= 3:
                        logging.warning(
                            "Hard secondary constraint (max_deviation=%.4f) left only %d/%d candidates; "
                            "the plot shows all points, the selection uses only the feasible ones.",
                            secondary_max_dev,
                            nr_feasible,
                            len(feasible_mask_full),
                        )
                else:
                    logging.warning(
                        "Hard secondary constraint removed all candidates for metric '%s'. "
                        "Falling back to unconstrained selection.",
                        secondary_tuning_metric,
                    )
                    secondary_mode = "none"
            elif secondary_mode == "soft":
                excess_degradation = df_plot_stats["__secondary_excess_degradation"].astype(float)
                if tune_metric_take_max:
                    df_plot_stats["__selection_objective"] = (
                        df_plot_stats[primary_metric_col].astype(float)
                        - secondary_soft_lambda * excess_degradation
                    )
                else:
                    df_plot_stats["__selection_objective"] = (
                        df_plot_stats[primary_metric_col].astype(float)
                        + secondary_soft_lambda * excess_degradation
                    )
                selection_metric_col = "__selection_objective"
                print(
                    f"Applied soft secondary constraint on '{secondary_tuning_metric}' with "
                    f"max_deviation={secondary_max_dev} and penalty_lambda={secondary_soft_lambda}."
                )

        def _should_log(values: pd.Series) -> bool:
            if not pd.api.types.is_numeric_dtype(values):
                return False
            v_min = values.min()
            v_max = values.max()
            if v_min <= 0 or v_max <= 0:
                return False
            ratio = v_max / v_min
            return ratio > math.pow(10, log_lin_scale_threshold)

        # from now on we assume that after aggregation we have a unique metric depending on at most two tuning parameters
        if selection_method == "best_overall":
            best_idx = df_plot_stats[selection_metric_col].idxmin() if not tune_metric_take_max else df_plot_stats[selection_metric_col].idxmax()
            best_row = df_plot_stats.loc[best_idx]
            for param in sweep_tune_params:
                ret.append(f"{param.replace('tun:','')}" + "=" + str(best_row[param]))
        elif selection_method == "fit_spline_take_best_overall":
            from scipy.interpolate import UnivariateSpline, SmoothBivariateSpline

            if len(sweep_tune_params) == 1:
                param = sweep_tune_params[0]
                x = df_plot_stats[param].values
                y = df_plot_stats[selection_metric_col].values

                use_log_x = log_lin_scale_auto_enabled and apply_log_to_spline and _should_log(df_plot_stats[param])
                if use_log_x:
                    x_fit = np.log10(x.astype(float))
                else:
                    x_fit = x.astype(float)

                # Fit a univariate spline to the data
                spline = UnivariateSpline(x_fit, y, s=spline_smoothing_factor)
                x_spline = np.linspace(x_fit.min(), x_fit.max(), spline_nr_points)
                y_spline = spline(x_spline)

                best_idx = np.argmin(y_spline) if not tune_metric_take_max else np.argmax(y_spline)
                best_param_value = float(np.power(10, x_spline[best_idx])) if use_log_x else x_spline[best_idx]
                ret.append(f"{param.replace('tun:','')}" + "=" + str(best_param_value))
                spline_plot_data = {
                    "mode": "1d",
                    "param": param,
                    "use_log_x": use_log_x,
                    "x_spline": x_spline,
                    "y_spline": y_spline,
                }
            elif len(sweep_tune_params) == 2:
                param_x = sweep_tune_params[0]
                param_y = sweep_tune_params[1]

                if not pd.api.types.is_numeric_dtype(df_plot_stats[param_x]) or not pd.api.types.is_numeric_dtype(df_plot_stats[param_y]):
                    raise NotImplementedError("2D spline fitting requires numeric tuning parameters.")

                x = df_plot_stats[param_x].astype(float).values
                y = df_plot_stats[param_y].astype(float).values
                z = df_plot_stats[selection_metric_col].values

                use_log_x = log_lin_scale_auto_enabled and apply_log_to_spline and _should_log(df_plot_stats[param_x])
                use_log_y = log_lin_scale_auto_enabled and apply_log_to_spline and _should_log(df_plot_stats[param_y])
                if use_log_x:
                    x_fit = np.log10(x)
                else:
                    x_fit = x
                if use_log_y:
                    y_fit = np.log10(y)
                else:
                    y_fit = y

                x_unique = np.unique(x)
                y_unique = np.unique(y)
                kx = min(3, len(x_unique) - 1)
                ky = min(3, len(y_unique) - 1)
                if kx < 1 or ky < 1:
                    logging.warning("Not enough unique points for 2D spline; falling back to best_overall on grid values.")
                    best_idx = np.argmin(z) if not tune_metric_take_max else np.argmax(z)
                    ret.append(f"{param_x.replace('tun:','')}" + "=" + str(x[best_idx]))
                    ret.append(f"{param_y.replace('tun:','')}" + "=" + str(y[best_idx]))
                    spline_plot_data = {
                        "mode": "2d_fallback",
                        "param_x": param_x,
                        "param_y": param_y,
                        "x": x,
                        "y": y,
                        "z": z,
                    }
                    spline = None
                else:
                    spline = SmoothBivariateSpline(x_fit, y_fit, z, s=spline_smoothing_factor, kx=kx, ky=ky)

                if spline is not None:
                    x_grid = np.linspace(x_fit.min(), x_fit.max(), spline_nr_points)
                    y_grid = np.linspace(y_fit.min(), y_fit.max(), spline_nr_points)
                    z_grid = spline(x_grid, y_grid)

                    best_idx = np.unravel_index(
                        np.argmin(z_grid) if not tune_metric_take_max else np.argmax(z_grid),
                        z_grid.shape,
                    )
                    best_x = x_grid[best_idx[0]]
                    best_y = y_grid[best_idx[1]]
                    best_x = float(np.power(10, best_x)) if use_log_x else best_x
                    best_y = float(np.power(10, best_y)) if use_log_y else best_y
                    ret.append(f"{param_x.replace('tun:','')}" + "=" + str(best_x))
                    ret.append(f"{param_y.replace('tun:','')}" + "=" + str(best_y))

                    spline_plot_data = {
                        "mode": "2d",
                        "param_x": param_x,
                        "param_y": param_y,
                        "use_log_x": use_log_x,
                        "use_log_y": use_log_y,
                        "x_grid": x_grid,
                        "y_grid": y_grid,
                        "z_grid": z_grid,
                        "best_idx": best_idx,
                        "best_x": best_x,
                        "best_y": best_y,
                        "x_fit": x_fit,
                        "y_fit": y_fit,
                        "z_fit": z,
                    }
            else:
                raise NotImplementedError("Spline fitting for more than two tuning parameters is not implemented yet.")
        else:
            raise ValueError(f"Unknown selection_method '{selection_method}'")

        if fail_on_boundary_best:
            for param in sweep_tune_params:
                if f"{param.replace('tun:','')}" not in " ".join(ret):
                    continue
                sel_val = None
                for override in ret:
                    key, value = override.split("=", 1)
                    if key == param.replace("tun:", ""):
                        try:
                            sel_val = float(value)
                        except ValueError:
                            sel_val = value
                        break
                if sel_val is None:
                    continue
                series = df_plot_stats[param]
                if pd.api.types.is_numeric_dtype(series) and isinstance(sel_val, (int, float, np.floating)):
                    vmin = float(series.min())
                    vmax = float(series.max())
                    if np.isclose(sel_val, vmin) or np.isclose(sel_val, vmax):
                        raise ValueError(f"Selected {param}={sel_val} lies on boundary [{vmin}, {vmax}]. Expand sweep range.")
                else:
                    if str(sel_val) == str(series.min()) or str(sel_val) == str(series.max()):
                        raise ValueError(f"Selected {param}={sel_val} lies on boundary of categorical sweep. Expand sweep range.")

        if add_wandb_plot:
            # Parse selected tuning values from ret entries like "++param.path=0.1" or "param.path=0.1"
            selected_values: Dict[str, Any] = {}
            for override in ret:
                if "=" not in override:
                    continue
                key_value = override # override[2:] if override.startswith("++") else override
                key, value = key_value.split("=", 1)
                tun_key = f"tun:{key}"
                try:
                    selected_values[tun_key] = float(value)
                except ValueError:
                    selected_values[tun_key] = value

            print(f"Selected tuning values for plotting: {selected_values}")

            def _try_float(value: Any) -> Optional[float]:
                try:
                    return float(value)
                except Exception:
                    return None

            def _find_idx(vals, target):
                target_num = _try_float(target)
                for i, v in enumerate(vals):
                    v_num = _try_float(v)
                    if v_num is not None and target_num is not None:
                        if np.isclose(v_num, target_num):
                            return i
                    else:
                        if str(v).strip() == str(target).strip():
                            return i
                return None

            def _plot_2d_heatmap(
                data_frame: pd.DataFrame,
                value_col: str,
                title: str,
                fmt: str,
                log_key: str,
                cmap: str | ListedColormap = "viridis",
                cbar_kws: Optional[Dict[str, Any]] = None,
                vmin: Optional[float] = None,
                vmax: Optional[float] = None,
            ) -> None:
                df_plot_heat = data_frame.copy()
                use_log_x = log_lin_scale_auto_enabled and _should_log(data_frame[param_x])
                use_log_y = log_lin_scale_auto_enabled and _should_log(data_frame[param_y])
                if use_log_x:
                    df_plot_heat[param_x] = np.log10(df_plot_heat[param_x].astype(float))
                if use_log_y:
                    df_plot_heat[param_y] = np.log10(df_plot_heat[param_y].astype(float))

                pivot_table = df_plot_heat.pivot(index=param_y, columns=param_x, values=value_col)
                pivot_table = pivot_table.sort_index(axis=0).sort_index(axis=1)

                plt.figure(figsize=(10, 6), dpi=dpi)
                ax = sns.heatmap(
                    pivot_table,
                    annot=True,
                    fmt=fmt,
                    cmap=cmap,
                    cbar_kws=cbar_kws,
                    vmin=vmin,
                    vmax=vmax,
                )

                xlabel = f"log10({param_x})" if use_log_x else param_x
                ylabel = f"log10({param_y})" if use_log_y else param_y
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(title)

                xticks = np.linspace(0, len(pivot_table.columns) - 1, num=min(5, len(pivot_table.columns)))
                yticks = np.linspace(0, len(pivot_table.index) - 1, num=min(5, len(pivot_table.index)))
                ax.set_xticks(xticks)
                ax.set_yticks(yticks)

                x_vals = np.interp(xticks, np.arange(len(pivot_table.columns)), pivot_table.columns.values.astype(float))
                y_vals = np.interp(yticks, np.arange(len(pivot_table.index)), pivot_table.index.values.astype(float))
                if use_log_x:
                    x_vals = np.power(10, x_vals)
                if use_log_y:
                    y_vals = np.power(10, y_vals)
                ax.set_xticklabels([f"{v:.2g}" for v in x_vals])
                ax.set_yticklabels([f"{v:.2g}" for v in y_vals])

                if param_x in selected_values and param_y in selected_values:
                    x_sel = selected_values[param_x]
                    y_sel = selected_values[param_y]

                    if use_log_x and _try_float(x_sel) is not None:
                        x_sel = np.log10(float(x_sel))
                    if use_log_y and _try_float(y_sel) is not None:
                        y_sel = np.log10(float(y_sel))

                    x_idx = _find_idx(list(pivot_table.columns), x_sel)
                    y_idx = _find_idx(list(pivot_table.index), y_sel)

                    if x_idx is not None and y_idx is not None:
                        ax.scatter(x_idx + 0.5, y_idx + 0.5, color="red", s=150, marker="*", label="Selected", zorder=5)
                        ax.legend(loc="upper right")

                plt.tight_layout()
                wandb.log({log_key: wandb.Image(plt)})
                plt.close()

            if len(sweep_tune_params) == 1:
                # Use full data for visualization (before secondary constraint filtering)
                # to show the complete exploration landscape
                df_plot_1d = df_plot_stats_full if secondary_mode != "none" else df_plot_stats
                param = sweep_tune_params[0]

                def _plot_1d_metric(
                    data_frame: pd.DataFrame,
                    value_col: str,
                    std_col: str,
                    ylabel: str,
                    title: str,
                    log_key: str,
                    include_spline: bool = False,
                ) -> None:
                    plt.figure(figsize=(10, 6), dpi=dpi)
                    x = data_frame[param].values
                    y = data_frame[value_col].values
                    yerr = data_frame[std_col].fillna(0.0).values
                    x_num = _try_float(x[0]) if len(x) > 0 else None

                    if x_num is not None:
                        x_numeric = np.asarray(x, dtype=float)
                        y_numeric = np.asarray(y, dtype=float)
                        yerr_numeric = np.asarray(yerr, dtype=float)
                        sort_idx = np.argsort(x_numeric)
                        x_sorted = x_numeric[sort_idx]
                        y_sorted = y_numeric[sort_idx]
                        yerr_sorted = yerr_numeric[sort_idx]

                        if secondary_mode != "none" and "__secondary_feasible" in data_frame.columns:
                            feasible_sorted = data_frame.iloc[sort_idx]["__secondary_feasible"].astype(bool).values
                            feasible_mask = feasible_sorted
                            if feasible_mask.any():
                                plt.plot(
                                    x_sorted[feasible_mask],
                                    y_sorted[feasible_mask],
                                    color="C2",
                                    marker="o",
                                    linestyle="-",
                                    label="Feasible",
                                    linewidth=2,
                                    markersize=6,
                                )
                                plt.fill_between(
                                    x_sorted[feasible_mask],
                                    y_sorted[feasible_mask] - yerr_sorted[feasible_mask],
                                    y_sorted[feasible_mask] + yerr_sorted[feasible_mask],
                                    color="C2",
                                    alpha=0.2,
                                )
                            infeasible_mask = ~feasible_sorted
                            if infeasible_mask.any():
                                plt.scatter(
                                    x_sorted[infeasible_mask],
                                    y_sorted[infeasible_mask],
                                    color="C3",
                                    marker="x",
                                    s=100,
                                    label="Infeasible (sec. constraint)",
                                    zorder=3,
                                    linewidths=2,
                                )
                        else:
                            plt.plot(x_sorted, y_sorted, color="C0", label="Mean")
                            plt.fill_between(
                                x_sorted,
                                y_sorted - yerr_sorted,
                                y_sorted + yerr_sorted,
                                color="C0",
                                alpha=0.2,
                                label="Std",
                            )
                    else:
                        x_strings = np.asarray(x, dtype=str)
                        y_numeric = np.asarray(y, dtype=float)
                        yerr_numeric = np.asarray(yerr, dtype=float)
                        order = np.argsort(x_strings)
                        x_sorted = x_strings[order]
                        y_sorted = y_numeric[order]
                        yerr_sorted = yerr_numeric[order]
                        x_pos = np.arange(len(x_sorted))
                        plt.plot(x_pos, y_sorted, color="C0", label="Mean")
                        plt.fill_between(
                            x_pos,
                            y_sorted - yerr_sorted,
                            y_sorted + yerr_sorted,
                            color="C0",
                            alpha=0.2,
                            label="Std",
                        )
                        plt.xticks(x_pos, x_sorted.tolist())

                    if log_lin_scale_auto_enabled and pd.api.types.is_numeric_dtype(data_frame[param]):
                        x_min = data_frame[param].min()
                        x_max = data_frame[param].max()
                        if x_min > 0 and x_max > 0:
                            x_ratio = x_max / x_min
                            if x_ratio > math.pow(10, log_lin_scale_threshold):
                                plt.xscale('log')

                    if include_spline and selection_method == "fit_spline_take_best_overall" and spline_plot_data is not None:
                        sp = spline_plot_data
                        if sp.get("mode") == "1d" and sp["param"] == param:
                            if sp["use_log_x"]:
                                x_plot = np.power(10, sp["x_spline"])
                            else:
                                x_plot = sp["x_spline"]
                            plt.plot(x_plot, sp["y_spline"], color="C1", linestyle="--", label="Spline fit")

                    if log_lin_scale_auto_enabled:
                        y_numeric = np.asarray(y, dtype=float)
                        y_min = float(np.nanmin(y_numeric))
                        y_max = float(np.nanmax(y_numeric))
                        if y_min > 0 and y_max > 0:
                            y_ratio = y_max / y_min
                            if y_ratio > math.pow(10, log_lin_scale_threshold):
                                plt.yscale('log')

                    if param in selected_values:
                        x_sel = selected_values[param]
                        x_sel_num = _try_float(x_sel)
                        if x_sel_num is not None and x_num is not None:
                            selected_match = data_frame[np.isclose(data_frame[param].astype(float), x_sel_num)]
                            if not selected_match.empty:
                                y_sel = float(selected_match.iloc[0][value_col])
                            else:
                                y_sel = float(np.interp(float(x_sel), x_sorted, y_sorted))
                            plt.scatter(
                                [x_sel_num],
                                [y_sel],
                                color="red",
                                s=100,
                                marker="*",
                                label=f"Sel.({float(x_sel):.2g})",
                                zorder=5,
                            )
                        else:
                            match = data_frame[data_frame[param].astype(str) == str(x_sel)]
                            if not match.empty:
                                y_sel = match.iloc[0][value_col]
                                idx = _find_idx(list(x_sorted), str(x_sel))
                                if idx is not None:
                                    plt.scatter(
                                        [idx],
                                        [y_sel],
                                        color="red",
                                        s=100,
                                        marker="*",
                                        label=f"Sel.({x_sel})",
                                        zorder=5,
                                    )

                    plt.xlabel(param)
                    plt.ylabel(ylabel)
                    plt.title(title)
                    plt.legend()
                    plt.tight_layout()
                    wandb.log({log_key: wandb.Image(plt)})
                    plt.close()

                _plot_1d_metric(
                    data_frame=df_plot_1d,
                    value_col=metric_mean_cols[tuning_metric],
                    std_col=metric_std_cols[tuning_metric],
                    ylabel=tuning_metric,
                    title=_phase_title(f"{tuning_metric} vs {param}"),
                    log_key="tuning_selection_plot",
                    include_spline=True,
                )

                if secondary_tuning_metric is not None and secondary_metric_col is not None:
                    _plot_1d_metric(
                        data_frame=df_plot_stats_full,
                        value_col=secondary_metric_col,
                        std_col=metric_std_cols[secondary_tuning_metric],
                        ylabel=secondary_tuning_metric,
                        title=_phase_title(f"{secondary_tuning_metric} vs {param}"),
                        log_key="tuning_selection_secondary_plot",
                        include_spline=False,
                    )

            elif len(sweep_tune_params) == 2:
                param_x = sweep_tune_params[0]
                param_y = sweep_tune_params[1]
                _plot_2d_heatmap(
                    data_frame=df_plot_stats_full,
                    value_col=metric_mean_cols[tuning_metric],
                    title=_phase_title(f"{tuning_metric} heatmap"),
                    fmt=".2f",
                    log_key="tuning_selection_primary_heatmap",
                )

                if secondary_tuning_metric is not None and secondary_metric_col is not None:
                    _plot_2d_heatmap(
                        data_frame=df_plot_stats_full,
                        value_col=secondary_metric_col,
                        title=_phase_title(f"{secondary_tuning_metric} heatmap"),
                        fmt=".3f",
                        log_key="tuning_selection_secondary_heatmap",
                    )

                if secondary_mode != "none" and secondary_tuning_metric is not None:
                    status_df = df_plot_stats_full.copy()
                    status_df["__secondary_status"] = np.where(
                        status_df["__secondary_feasible"].astype(bool),
                        1,
                        0,
                    )

                    if param_x in selected_values and param_y in selected_values:
                        x_selected_num = _try_float(selected_values[param_x])
                        y_selected_num = _try_float(selected_values[param_y])
                        if x_selected_num is not None and y_selected_num is not None:
                            selected_mask = (
                                np.isclose(status_df[param_x].astype(float), x_selected_num)
                                & np.isclose(status_df[param_y].astype(float), y_selected_num)
                            )
                        else:
                            selected_mask = (
                                status_df[param_x].astype(str) == str(selected_values[param_x])
                            ) & (
                                status_df[param_y].astype(str) == str(selected_values[param_y])
                            )
                        status_df.loc[selected_mask, "__secondary_status"] = 2

                    _plot_2d_heatmap(
                        data_frame=status_df,
                        value_col="__secondary_status",
                        title=_phase_title(f"Secondary feasibility/selection ({secondary_mode})"),
                        fmt=".0f",
                        log_key="tuning_selection_secondary_feasibility_heatmap",
                        cmap=ListedColormap(["#d62728", "#2ca02c", "#1f77b4"]),
                        cbar_kws={"ticks": [0, 1, 2], "label": "0=infeasible, 1=feasible, 2=selected"},
                        vmin=0,
                        vmax=2,
                    )

                if selection_method == "fit_spline_take_best_overall" and spline_plot_data is not None:
                    sp = spline_plot_data
                    if sp.get("mode") == "2d":
                        plt.figure(figsize=(10, 6), dpi=dpi)
                        # z_grid shape from spline(x_grid, y_grid) is (len_x, len_y)
                        # pcolormesh expects C with shape (len_y, len_x), so transpose
                        z_grid = sp["z_grid"].T
                        x_grid = sp["x_grid"]  # in log10 space if use_log_x
                        y_grid = sp["y_grid"]  # in log10 space if use_log_y

                        # Convert grid back to original scale for plotting with native log axes
                        x_plot = np.power(10, x_grid) if sp["use_log_x"] else x_grid
                        y_plot = np.power(10, y_grid) if sp["use_log_y"] else y_grid

                        ax_spline = plt.gca()
                        X, Y = np.meshgrid(x_plot, y_plot)
                        im = ax_spline.pcolormesh(X, Y, z_grid, shading='gouraud', cmap='viridis')
                        plt.colorbar(im, ax=ax_spline, label=tuning_metric)

                        # Use matplotlib native log scale (matches seaborn heatmap visual)
                        if sp["use_log_x"]:
                            ax_spline.set_xscale('log')
                        if sp["use_log_y"]:
                            ax_spline.set_yscale('log')
                        # Invert y to match seaborn heatmap (smallest at top)
                        ax_spline.invert_yaxis()

                        ax_spline.set_xlabel(sp["param_x"])
                        ax_spline.set_ylabel(sp["param_y"])
                        ax_spline.set_title(_phase_title(f"{tuning_metric} spline surface"))

                        # Mark selected tuning point (in original scale)
                        if param_x in selected_values and param_y in selected_values:
                            x_sel = selected_values[param_x]
                            y_sel = selected_values[param_y]
                            x_sel_num = _try_float(x_sel)
                            y_sel_num = _try_float(y_sel)
                            if x_sel_num is not None and y_sel_num is not None:
                                ax_spline.scatter(x_sel_num, y_sel_num, color="red", s=150, marker="*", label="Selected", zorder=5)
                                ax_spline.legend(loc="upper right")
                                if "best_x" in sp and "best_y" in sp:
                                    best_x_num = _try_float(sp["best_x"])
                                    best_y_num = _try_float(sp["best_y"])
                                    x_sel_num2 = _try_float(x_sel)
                                    y_sel_num2 = _try_float(y_sel)
                                    dx = abs(best_x_num - x_sel_num2) if best_x_num is not None and x_sel_num2 is not None else None
                                    dy = abs(best_y_num - y_sel_num2) if best_y_num is not None and y_sel_num2 is not None else None
                                    if dx is not None and dy is not None and (dx > 1e-6 or dy > 1e-6):
                                        logging.warning("Selected values differ from spline optimum: sel=(%s,%s) best=(%s,%s)", x_sel, y_sel, sp["best_x"], sp["best_y"])
                        # Overlay measured points in original scale
                        x_pts = df_plot_stats[param_x].astype(float).values
                        y_pts = df_plot_stats[param_y].astype(float).values
                        z_pts = df_plot_stats[metric_mean_cols[tuning_metric]].values
                        mask = np.isfinite(x_pts) & np.isfinite(y_pts) & (x_pts > 0) & (y_pts > 0)
                        x_pts = x_pts[mask]
                        y_pts = y_pts[mask]
                        z_pts = z_pts[mask]
                        if x_pts.size > 0 and y_pts.size > 0:
                            ax_spline.scatter(
                                x_pts,
                                y_pts,
                                c=z_pts,
                                cmap='viridis',
                                vmin=z_grid.min(),
                                vmax=z_grid.max(),
                                s=60,
                                edgecolors="white",
                                linewidths=1.0,
                                zorder=4,
                                label="Measured",
                            )
                            # Annotate measured z values for visual verification of spline fit
                            for xi, yi, zi in zip(x_pts, y_pts, z_pts):
                                ax_spline.annotate(f"{zi:.2f}", (xi, yi), fontsize=6,
                                                   color="white", ha="left", va="bottom",
                                                   xytext=(3, 3), textcoords="offset points",
                                                   bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))
                        plt.tight_layout()
                        wandb.log({"tuning_selection_spline_plot": wandb.Image(plt)})
                        plt.close()

            elif len(sweep_tune_params) >= 3:
                # Pairwise 2D heatmaps: for each pair (p_i, p_j), marginalize the
                # remaining parameters by taking the best (max/min) metric value.
                from itertools import combinations

                agg_fn = "max" if tune_metric_take_max else "min"

                for param_x, param_y in combinations(sweep_tune_params, 2):
                    other_params = [p for p in sweep_tune_params if p not in (param_x, param_y)]

                    # Build a reduced dataframe: best metric over the other dims for each (param_x, param_y) pair
                    cols_needed = [param_x, param_y] + list(metric_mean_cols.values())
                    if secondary_metric_col is not None and secondary_metric_col in df_plot_stats_full.columns:
                        cols_needed.append(secondary_metric_col)
                    cols_needed = list(dict.fromkeys(c for c in cols_needed if c in df_plot_stats_full.columns))
                    df_marginal = (
                        df_plot_stats_full[cols_needed]
                        .groupby([param_x, param_y], dropna=False)
                        .agg(agg_fn)
                        .reset_index()
                    )

                    pair_label = f"{param_x.replace('tun:','')} vs {param_y.replace('tun:','')}"

                    _plot_2d_heatmap(
                        data_frame=df_marginal,
                        value_col=metric_mean_cols[tuning_metric],
                        title=_phase_title(f"{tuning_metric} heatmap ({pair_label}, others={agg_fn})"),
                        fmt=".2f",
                        log_key=f"tuning_selection_primary_heatmap_{param_x.replace('tun:','').replace('++','')}_{param_y.replace('tun:','').replace('++','')}",
                    )

                    if secondary_tuning_metric is not None and secondary_metric_col is not None and secondary_metric_col in df_marginal.columns:
                        _plot_2d_heatmap(
                            data_frame=df_marginal,
                            value_col=secondary_metric_col,
                            title=_phase_title(f"{secondary_tuning_metric} heatmap ({pair_label}, others={agg_fn})"),
                            fmt=".3f",
                            log_key=f"tuning_selection_secondary_heatmap_{param_x.replace('tun:','').replace('++','')}_{param_y.replace('tun:','').replace('++','')}",
                        )

        print(f"Selected hyperparameters: {ret}")

        # ── Phase-1 fallback check ────────────────────────────────────────────
        # If this is a phase-2 selection and phase-1 actually produced a better
        # result than anything in the phase-2 neighborhood, fall back to phase-1.
        if (
            phase1_reference_results is not None
            and phase1_reference_df_sweep is not None
            and phase1_reference_best_overrides is not None
            and len(ret) > 0
        ):
            try:
                df_ph1_results = pd.DataFrame(
                    [_result_to_dict(r) for r in phase1_reference_results]
                )
                ph1_tune_params = [
                    c for c in phase1_reference_df_sweep.columns if c.startswith("tun:")
                ]
                if len(ph1_tune_params) > 0 and tuning_metric in df_ph1_results.columns:
                    df_ph1_plot = pd.concat(
                        [df_ph1_results[[tuning_metric]], phase1_reference_df_sweep],
                        axis=1,
                    )
                    ph1_stats = (
                        df_ph1_plot.groupby(ph1_tune_params, dropna=False)[tuning_metric]
                        .mean()
                        .reset_index()
                    )
                    ph1_best_metric = (
                        float(ph1_stats[tuning_metric].max())
                        if tune_metric_take_max
                        else float(ph1_stats[tuning_metric].min())
                    )

                    # Phase-2 best metric (from the already-aggregated stats)
                    ph2_best_metric = (
                        float(df_plot_stats[primary_metric_col].max())
                        if tune_metric_take_max
                        else float(df_plot_stats[primary_metric_col].min())
                    )

                    phase1_better = (
                        ph1_best_metric > ph2_best_metric
                        if tune_metric_take_max
                        else ph1_best_metric < ph2_best_metric
                    )

                    wandb.log({
                        "phase1_vs_phase2/phase1_best_metric": ph1_best_metric,
                        "phase1_vs_phase2/phase2_best_metric": ph2_best_metric,
                        "phase1_vs_phase2/fallback_triggered": int(phase1_better),
                    })

                    if phase1_better:
                        logging.warning(
                            "Phase-2 refinement did not improve over phase-1 "
                            "(%s: phase1=%.6f  phase2=%.6f). "
                            "Falling back to phase-1 best overrides: %s",
                            tuning_metric, ph1_best_metric, ph2_best_metric,
                            phase1_reference_best_overrides,
                        )
                        # try:
                            # wandb.alert(
                                # title="Phase-2 fallback to phase-1",
                                # text=(
                                    # f"Phase-2 refinement did not improve: "
                                    # f"phase1 {tuning_metric}={ph1_best_metric:.6f} is better than "
                                    # f"phase2 {tuning_metric}={ph2_best_metric:.6f}. "
                                    # f"Using phase-1 best: {phase1_reference_best_overrides}"
                                # ),
                            # )
                        # except Exception:
                            # pass
                        ret = list(phase1_reference_best_overrides)
                        print(f"[Fallback] Using phase-1 hyperparameters: {ret}")
                    else:
                        logging.info(
                            "Phase-2 improved over phase-1 (%s: phase1=%.6f → phase2=%.6f). "
                            "Keeping phase-2 selection.",
                            tuning_metric, ph1_best_metric, ph2_best_metric,
                        )
            except Exception as _fallback_err:
                logging.warning(
                    "Phase-1 fallback check failed (%s). Keeping phase-2 selection.",
                    _fallback_err,
                )
        # ─────────────────────────────────────────────────────────────────────

    return CacheableList(cfg=ret)
