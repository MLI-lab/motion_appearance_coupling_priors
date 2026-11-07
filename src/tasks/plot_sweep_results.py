from prefect import flow, task
from prefect.cache_policies import TASK_SOURCE, INPUTS, NONE
import logging
import sys
import wandb
from src.utils.wandb_utils import wandb_kwargs_for_prefect_task, WandbParamsTask
from typing import Any, List, Optional, Dict, Sequence, cast
from src.prefect.caching import CacheableDict, CacheableDictConfig, CacheableListConfig
import pandas as pd
import seaborn as sns
from typing import Tuple
import io
import math
import re
import textwrap
import numpy as np
from itertools import permutations
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
from src.prefect.setup import StoragePath
from omegaconf import OmegaConf


def _to_jsonable_scalar(value):
    if isinstance(value, pd.Series):
        return ", ".join(str(v) for v in value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, list, tuple, set, dict)):
        return str(value)
    if pd.isna(value):
        return None
    return value


def _result_to_dict(result: Any) -> Dict[str, Any]:
    if isinstance(result, tuple) and len(result) == 2:
        metrics, storage = result
        result_dict = _result_to_dict(metrics)
        if isinstance(storage, StoragePath):
            result_dict["storage_path"] = storage.model_dump(mode="python")
        elif isinstance(storage, dict):
            result_dict["storage_path"] = storage
        return result_dict
    if isinstance(result, CacheableDict):
        return dict(result.cfg)
    if isinstance(result, CacheableDictConfig):
        container = OmegaConf.to_container(result.cfg, resolve=True)
        return cast(Dict[str, Any], container) if isinstance(container, dict) else {"value": container}
    if isinstance(result, StoragePath):
        return {
            "storage_path": result.model_dump(mode="python"),
        }
    if isinstance(result, dict):
        return dict(result)
    cfg_value = getattr(result, "cfg", None)
    if cfg_value is not None and not isinstance(result, tuple):
        return dict(cfg_value)
    raise TypeError(f"Unsupported result type for plotting: {type(result)}")


def _sanitize_table_df(df: pd.DataFrame) -> pd.DataFrame:
    sanitized_df = df.copy()
    for col in sanitized_df.columns:
        sanitized_df[col] = sanitized_df[col].map(lambda value: _to_jsonable_scalar(value))
    return sanitized_df


def _normalize_param_name(param_name: str) -> str:
    return str(param_name).strip().strip('"').strip("'").lstrip("+")


def _resolve_column_name(requested_name: str, available_columns: List[str]) -> Optional[str]:
    if requested_name in available_columns:
        return requested_name

    normalized_requested = _normalize_param_name(requested_name)
    normalized_to_actual = {
        _normalize_param_name(col): col for col in available_columns
    }
    return normalized_to_actual.get(normalized_requested)


def _filter_metrics_by_cfg(
    metrics: List[str],
    metrics_filter_cfg: Optional[CacheableListConfig],
    warning_context: str,
) -> List[str]:
    if metrics_filter_cfg is None:
        return metrics

    metric_filters = [str(v).strip().lower() for v in metrics_filter_cfg.cfg if str(v).strip()]
    if len(metric_filters) == 0:
        return metrics

    filtered_metrics = [
        metric for metric in metrics
        if any(metric_filter in metric.lower() for metric_filter in metric_filters)
    ]
    if len(filtered_metrics) == 0:
        logging.warning(
            f"No metrics matched filters {metric_filters} for {warning_context}. Falling back to all numeric metrics."
        )
        return metrics
    return filtered_metrics


def _infer_metric_higher_is_better(metric_name: str) -> bool:
    metric_name = metric_name.lower()
    # "epe"/"outlier": optical-flow metrics; "runtime"/"memory": compute-cost metrics
    lower_is_better_keywords = ["loss", "error", "mse", "mae", "rmse", "nll", "lpips", "epe", "outlier", "runtime", "memory"]
    return not any(keyword in metric_name for keyword in lower_is_better_keywords)


def _order_metrics_by_filter(
    metrics: List[str],
    metrics_filter_cfg: Optional[CacheableListConfig],
) -> List[str]:
    """
    Order metrics by the position of their matching filter term in the config;
    unmatched metrics go last in their original order.
    """
    if metrics_filter_cfg is None:
        return metrics
    metric_filters = [str(v).strip().lower() for v in metrics_filter_cfg.cfg if str(v).strip()]
    if len(metric_filters) == 0:
        return metrics

    def sort_key(metric: str) -> Tuple[int, int]:
        for filter_idx, metric_filter in enumerate(metric_filters):
            if metric_filter in metric.lower():
                return (filter_idx, metrics.index(metric))
        return (len(metric_filters), metrics.index(metric))

    return sorted(metrics, key=sort_key)


def _config_list_to_str_list(cfg: Optional[CacheableListConfig]) -> List[str]:
    if cfg is None:
        return []
    return [str(v) for v in cfg.cfg if str(v).strip()]


def _config_dict_to_str_map(cfg: Optional[CacheableDictConfig]) -> Dict[str, str]:
    if cfg is None:
        return {}
    container = OmegaConf.to_container(cfg.cfg, resolve=True)
    if not isinstance(container, dict):
        return {}
    return {str(key): str(value) for key, value in container.items()}


# unit suffixes for reported deltas (explicit paper_significance_metric_labels take precedence)
_METRIC_UNIT_SUFFIX = {"psnr": " dB", "snr": " dB", "ssnr": " dB", "runtime_s": " s", "memory_gb": " GB", "memory_reserved_gb": " GB"}


def _metric_unit_suffix(metric_name: str) -> str:
    lowered = metric_name.lower()
    for keyword, suffix in _METRIC_UNIT_SUFFIX.items():
        if keyword in lowered:
            return suffix
    return ""


def _delta_decimals(scale: float) -> int:
    """Decimal places giving ~3 significant digits for a delta of magnitude `scale`."""
    if not np.isfinite(scale) or scale <= 0:
        return 4
    return int(np.clip(2 - math.floor(math.log10(scale)), 2, 6))


def _format_signed(value: float, decimals: int, suffix: str = "") -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:+.{decimals}f}{suffix}"


def _format_pvalue(pvalue: Optional[float]) -> str:
    if pvalue is None or not np.isfinite(pvalue):
        return "n/a"
    if pvalue < 1e-3:
        return f"{pvalue:.1e}"
    return f"{pvalue:.3f}"


def _holm_adjust(pvalues: List[Optional[float]]) -> List[Optional[float]]:
    """Holm-Bonferroni step-down adjustment over the finite p-values in `pvalues`."""
    finite_idxs = [idx for idx, p in enumerate(pvalues) if p is not None and np.isfinite(p)]
    adjusted: List[Optional[float]] = [None] * len(pvalues)
    num_tests = len(finite_idxs)
    running_max = 0.0
    for rank, idx in enumerate(sorted(finite_idxs, key=lambda i: cast(float, pvalues[i]))):
        running_max = max(running_max, min(1.0, (num_tests - rank) * cast(float, pvalues[idx])))
        adjusted[idx] = running_max
    return adjusted


def _paired_delta_ci(
    deltas: np.ndarray,
    ci_method: str,
    ci_level: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Tuple[float, float]:
    n = len(deltas)
    if n < 2:
        # undefined for a single observation
        return (float("nan"), float("nan"))

    alpha = 1.0 - ci_level
    if ci_method == "t":
        sem = float(np.std(deltas, ddof=1)) / math.sqrt(n)
        try:
            from scipy.stats import t as student_t

            crit = float(student_t.ppf(1.0 - alpha / 2.0, df=n - 1))
        except Exception:  # pragma: no cover - scipy is a hard dependency elsewhere in the repo
            logging.warning("scipy unavailable for the t critical value; falling back to a normal approximation.")
            crit = 1.959963984540054
        mean = float(np.mean(deltas))
        return (mean - crit * sem, mean + crit * sem)

    rng = np.random.default_rng(bootstrap_seed)
    resample_idx = rng.integers(0, n, size=(int(bootstrap_samples), n))
    boot_means = deltas[resample_idx].mean(axis=1)
    lo, hi = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (float(lo), float(hi))


def _wilcoxon_pvalue(oriented_deltas: np.ndarray, alternative: str) -> Optional[float]:
    """
    Wilcoxon signed-rank p-value on deltas oriented so that positive == primary is better.
    """
    if len(oriented_deltas) == 0 or np.allclose(oriented_deltas, 0.0):
        return None
    try:
        from scipy.stats import wilcoxon
    except Exception:  # pragma: no cover - scipy is a hard dependency elsewhere in the repo
        logging.warning("scipy is unavailable; Wilcoxon p-values will be reported as 'n/a'.")
        return None
    try:
        return float(wilcoxon(oriented_deltas, alternative=alternative).pvalue)
    except Exception as exc:
        logging.warning(f"Wilcoxon signed-rank test failed ({exc}); reporting 'n/a'.")
        return None


_ROLLOUT_STEP_COLUMN_RE = re.compile(r"^(?P<base_metric>.+)_frame_(?P<step>\d+)$")


def _detect_rollout_step_metrics(
    columns: List[str],
    rollout_step_metrics_cfg: Optional[CacheableListConfig],
) -> Dict[str, Dict[int, str]]:
    """
    Group columns named '<metric>_frame_<k>' as {base metric: {rollout step: column name}}.
    """
    requested = None
    if rollout_step_metrics_cfg is not None:
        requested = {str(v).strip().lower() for v in rollout_step_metrics_cfg.cfg if str(v).strip()}

    grouped: Dict[str, Dict[int, str]] = {}
    for col in columns:
        match = _ROLLOUT_STEP_COLUMN_RE.match(col)
        if not match:
            continue
        base_metric = match.group("base_metric")
        if requested is not None and base_metric.lower() not in requested:
            continue
        grouped.setdefault(base_metric, {})[int(match.group("step"))] = col
    return grouped


def _render_table_figure(
    table_df: pd.DataFrame,
    title: str,
    figsize: Sequence[float],
    dpi: int,
    auto_column_width: bool = False,
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axis("off")

    table = ax.table(
        cellText=table_df.astype(str).values,
        colLabels=[str(col) for col in table_df.columns],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)

    if auto_column_width:
        # needed for tables with long text columns (significance table)
        table.auto_set_column_width(col=list(range(len(table_df.columns))))

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    # wrap long titles so they aren't clipped
    if len(title) > 72:
        ax.set_title("\n".join(textwrap.wrap(title, width=72)), fontsize=10)
    else:
        ax.set_title(title)
    plt.tight_layout()
    return fig


def _build_summary_table(
    df_plot: pd.DataFrame,
    sweep_params: List[str],
    stochastic_params: List[str],
    numeric_metrics: List[str],
) -> pd.DataFrame:
    group_cols = [param for param in sweep_params if param not in stochastic_params]
    if group_cols:
        mean_df = df_plot.groupby(group_cols, dropna=False)[numeric_metrics].mean().reset_index()
        std_df = df_plot.groupby(group_cols, dropna=False)[numeric_metrics].std().reset_index()
        std_df = std_df.rename(columns={metric: f"{metric}__std" for metric in numeric_metrics})
        agg = mean_df.merge(std_df, on=group_cols, how="left")
    else:
        agg = pd.DataFrame({"setting": ["all"]})
        for metric in numeric_metrics:
            agg[metric] = [df_plot[metric].mean()]
            agg[f"{metric}__std"] = [df_plot[metric].std()]
        group_cols = ["setting"]

    rows = []
    for _, row in agg.iterrows():
        row_dict = {col: _to_jsonable_scalar(row[col]) for col in group_cols}
        for metric in numeric_metrics:
            mean_val = row[metric]
            std_val = row[f"{metric}__std"]
            row_dict[metric] = f"{mean_val:.4g} ± {0.0 if pd.isna(std_val) else std_val:.3g}"
        rows.append(row_dict)

    return _sanitize_table_df(pd.DataFrame(rows, columns=group_cols + numeric_metrics))


def _build_cost_quality_table(
    df_plot: pd.DataFrame,
    method_col: str,
    budget_cols: List[str],
    metrics: List[str],
) -> pd.DataFrame:
    """
    One row per (method, budget) with mean / std of each metric over the other sweep dimensions.
    The budget is the first non-null of budget_cols (recon iterations / baseline nfe share one axis).
    """
    df = df_plot.copy()
    df["budget"] = df[budget_cols].bfill(axis=1).iloc[:, 0]
    df = df.dropna(subset=["budget"])
    df["budget"] = df["budget"].astype(int)
    grouped = df.groupby([method_col, "budget"], dropna=False)
    agg = grouped[metrics].agg(["mean", "std"])
    agg.columns = [f"{metric}__{stat}" for metric, stat in agg.columns]
    agg["n"] = grouped.size()
    if "recon_gpu_name" in df.columns:
        agg["gpus"] = grouped["recon_gpu_name"].agg(lambda names: ", ".join(sorted({str(n) for n in names.dropna()})))
    return agg.reset_index().sort_values([method_col, "budget"]).reset_index(drop=True)


def _build_winrate_table(
    df_plot: pd.DataFrame,
    sweep_params: List[str],
    stochastic_params: List[str],
    numeric_metrics: List[str],
    winrate_over_param: str,
    higher_is_better_override: Optional[Dict[str, bool]] = None,
) -> Optional[pd.DataFrame]:
    if winrate_over_param not in df_plot.columns:
        logging.warning(f"Win-rate table skipped: parameter '{winrate_over_param}' not present in dataframe.")
        return None

    config_cols = [param for param in sweep_params if param not in stochastic_params]
    if not config_cols:
        config_cols = ["_all_config"]
        df_plot = df_plot.copy()
        df_plot["_all_config"] = "all"

    stochastic_other_cols = [
        param for param in stochastic_params if param in df_plot.columns and param != winrate_over_param
    ]

    pre_agg_cols = config_cols + [winrate_over_param]
    agg_group_cols = pre_agg_cols + stochastic_other_cols
    metric_agg = df_plot.groupby(agg_group_cols, dropna=False)[numeric_metrics].mean().reset_index()
    metric_agg = metric_agg.groupby(pre_agg_cols, dropna=False)[numeric_metrics].mean().reset_index()

    win_counts = {metric: { } for metric in numeric_metrics}
    total_buckets = metric_agg[winrate_over_param].nunique()
    if total_buckets == 0:
        return None

    for bucket_val, bucket_df in metric_agg.groupby(winrate_over_param, dropna=False):
        _ = bucket_val
        for metric in numeric_metrics:
            higher_is_better = (
                higher_is_better_override.get(metric)
                if higher_is_better_override and metric in higher_is_better_override
                else _infer_metric_higher_is_better(metric)
            )
            target_val = bucket_df[metric].max() if higher_is_better else bucket_df[metric].min()
            winners = bucket_df[np.isclose(bucket_df[metric], target_val, equal_nan=False)]
            if len(winners) == 0:
                continue
            shared_win = 1.0 / float(len(winners))
            for _, winner_row in winners.iterrows():
                key = tuple(winner_row[col] for col in config_cols)
                if key not in win_counts[metric]:
                    win_counts[metric][key] = 0.0
                win_counts[metric][key] += shared_win

    all_config_keys = sorted(
        {
            tuple(row[col] for col in config_cols)
            for _, row in metric_agg[config_cols].drop_duplicates().iterrows()
        }
    )

    output_rows = []
    for key in all_config_keys:
        row_dict = {col: key[idx] for idx, col in enumerate(config_cols)}
        for metric in numeric_metrics:
            wins = win_counts[metric].get(key, 0.0)
            row_dict[f"{metric} win-rate"] = f"{100.0 * wins / float(total_buckets):.2f}%"
        output_rows.append(row_dict)

    output_df = pd.DataFrame(output_rows, columns=config_cols + [f"{m} win-rate" for m in numeric_metrics])
    return _sanitize_table_df(output_df)


def _resolve_significance_pairing_cols(
    df_subset: pd.DataFrame,
    method_col: str,
    sweep_params: List[str],
    requested_pairing_params: List[str],
) -> List[str]:
    """
    Columns identifying one paired unit (default: all sweep params except the method).
    Columns with missing values are dropped, since they would yield zero matched pairs.
    """
    if requested_pairing_params:
        pairing_cols = []
        for param in requested_pairing_params:
            resolved = _resolve_column_name(param, df_subset.columns.tolist())
            if resolved is None:
                logging.warning(f"Significance table: pairing parameter '{param}' not found in dataframe; skipping it.")
            else:
                pairing_cols.append(resolved)
    else:
        pairing_cols = [param for param in sweep_params if param != method_col]

    populated_cols = []
    for col in pairing_cols:
        if df_subset[col].isna().any():
            logging.warning(
                f"Significance table: pairing parameter '{col}' is not populated for every compared run "
                f"(recon/baseline sweeps with differing keys); dropping it from the pairing key."
            )
            continue
        populated_cols.append(col)
    return populated_cols


def _build_significance_table(
    df_plot: pd.DataFrame,
    sweep_params: List[str],
    numeric_metrics: List[str],
    method_param: str,
    primary_value: str,
    requested_pairing_params: List[str] = [],
    requested_baselines: List[str] = [],
    method_labels: Dict[str, str] = {},
    metric_labels: Dict[str, str] = {},
    ci_method: str = "bootstrap",
    ci_level: float = 0.95,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 0,
    alternative: str = "two-sided",
    pvalue_correction: str = "none",
    higher_is_better_override: Optional[Dict[str, bool]] = None,
) -> Optional[Tuple[pd.DataFrame, str]]:
    """
    Paired significance table of the primary method vs. every other `method_param` value,
    one row per (metric, baseline): Mean Δ | CI | Win rate | Wilcoxon p | n.

    Mean Δ / CI are primary - baseline in raw units; win rate / Wilcoxon follow the metric's direction.
    Returns (table, title) or None.
    """
    method_col = _resolve_column_name(method_param, df_plot.columns.tolist())
    if method_col is None:
        logging.warning(
            f"Significance table skipped: method parameter '{method_param}' is not a column of the sweep dataframe "
            f"(available: {df_plot.columns.tolist()})."
        )
        return None

    method_values = [value for value in dict.fromkeys(df_plot[method_col].tolist()) if not pd.isna(value)]
    primary_matches = [value for value in method_values if str(value) == str(primary_value)]
    if not primary_matches:
        logging.warning(
            f"Significance table skipped: primary method '{primary_value}' not found in column '{method_col}' "
            f"(available: {[str(v) for v in method_values]})."
        )
        return None
    primary = primary_matches[0]

    if requested_baselines:
        baselines = []
        for requested in requested_baselines:
            matches = [value for value in method_values if str(value) == str(requested) and str(value) != str(primary)]
            if not matches:
                logging.warning(
                    f"Significance table: requested baseline '{requested}' not found in column '{method_col}'; skipping it."
                )
                continue
            baselines.append(matches[0])
    else:
        baselines = [value for value in method_values if str(value) != str(primary)]

    if not baselines:
        logging.warning("Significance table skipped: no baseline methods to compare the primary method against.")
        return None

    compared_values = [primary] + baselines
    df_subset = df_plot[df_plot[method_col].astype(str).isin([str(v) for v in compared_values])].copy()

    pairing_cols = _resolve_significance_pairing_cols(
        df_subset=df_subset,
        method_col=method_col,
        sweep_params=sweep_params,
        requested_pairing_params=requested_pairing_params,
    )
    if not pairing_cols:
        logging.warning(
            "Significance table skipped: no usable pairing parameter left, so there is nothing to pair the "
            "primary method and the baselines over (sweep over e.g. the sample index to enable this table)."
        )
        return None

    metrics = [metric for metric in numeric_metrics if metric in df_subset.columns]
    if not metrics:
        logging.warning("Significance table skipped: none of the requested metrics are present in the results.")
        return None

    # average out remaining dims (e.g. seeds): one value per (method, paired unit)
    agg = df_subset.groupby([method_col] + pairing_cols, dropna=False)[metrics].mean().reset_index()
    primary_df = agg[agg[method_col].astype(str) == str(primary)].set_index(pairing_cols)
    baseline_dfs = {
        str(baseline): agg[agg[method_col].astype(str) == str(baseline)].set_index(pairing_cols)
        for baseline in baselines
    }

    def _label_for(value: Any, labels: Dict[str, str]) -> str:
        return labels.get(str(value), str(value))

    rows: List[Dict[str, Any]] = []
    raw_pvalues: List[Optional[float]] = []
    for metric in metrics:
        higher_is_better = (
            higher_is_better_override.get(metric)
            if higher_is_better_override and metric in higher_is_better_override
            else _infer_metric_higher_is_better(metric)
        )
        metric_label = _label_for(metric, metric_labels)
        arrow = "↑" if higher_is_better else "↓"
        unit_suffix = _metric_unit_suffix(metric)

        metric_rows: List[Dict[str, Any]] = []
        scale = 0.0
        for baseline in baselines:
            baseline_df = baseline_dfs[str(baseline)]
            common_index = primary_df.index.intersection(baseline_df.index)
            paired = pd.DataFrame(
                {
                    "primary": primary_df.loc[common_index, metric],
                    "baseline": baseline_df.loc[common_index, metric],
                }
            ).dropna()
            deltas = (paired["primary"] - paired["baseline"]).to_numpy(dtype=float)
            n = len(deltas)
            if n == 0:
                logging.warning(
                    f"Significance table: no matched pairs for metric '{metric}' between '{primary}' and "
                    f"'{baseline}' over pairing key {pairing_cols}; reporting an empty row."
                )

            oriented = deltas if higher_is_better else -deltas
            mean_delta = float(np.mean(deltas)) if n > 0 else float("nan")
            ci_lo, ci_hi = _paired_delta_ci(
                deltas=deltas,
                ci_method=ci_method,
                ci_level=ci_level,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
            win_rate = (
                float(np.mean(np.where(oriented > 0, 1.0, np.where(oriented < 0, 0.0, 0.5)))) if n > 0 else float("nan")
            )
            pvalue = _wilcoxon_pvalue(oriented, alternative=alternative) if n > 0 else None

            candidate_scales = [
                abs(value) for value in (mean_delta, ci_hi - mean_delta, mean_delta - ci_lo) if np.isfinite(value)
            ]
            if candidate_scales:
                scale = max(scale, max(candidate_scales))
            metric_rows.append(
                {
                    "comparison": f"{metric_label} {arrow} vs. {_label_for(baseline, method_labels)}",
                    "mean_delta": mean_delta,
                    "ci": (ci_lo, ci_hi),
                    "win_rate": win_rate,
                    "n": n,
                }
            )
            raw_pvalues.append(pvalue)

        decimals = _delta_decimals(scale)
        for metric_row in metric_rows:
            ci_lo, ci_hi = metric_row["ci"]
            rows.append(
                {
                    "Comparison (ours - baseline)": metric_row["comparison"],
                    "Mean Δ": _format_signed(metric_row["mean_delta"], decimals, unit_suffix),
                    f"{ci_level:.0%} CI": (
                        f"[{_format_signed(ci_lo, decimals)}, {_format_signed(ci_hi, decimals)}]"
                        if np.isfinite(ci_lo) and np.isfinite(ci_hi)
                        else "n/a"
                    ),
                    "Win rate": f"{100.0 * metric_row['win_rate']:.0f}%" if np.isfinite(metric_row["win_rate"]) else "n/a",
                    "n": metric_row["n"],
                }
            )

    if not rows:
        return None

    table_df = pd.DataFrame(rows)
    table_df["Wilcoxon p"] = [_format_pvalue(p) for p in raw_pvalues]
    if str(pvalue_correction).lower() == "holm":
        table_df["Holm p"] = [_format_pvalue(p) for p in _holm_adjust(raw_pvalues)]
    # keep "n" last so the reported statistics stay in the requested column order
    table_df = table_df[[col for col in table_df.columns if col != "n"] + ["n"]]

    sample_sizes = sorted({int(row["n"]) for row in rows})
    n_str = str(sample_sizes[0]) if len(sample_sizes) == 1 else f"{sample_sizes[0]}-{sample_sizes[-1]}"
    if sample_sizes[-1] < 2:
        logging.warning(
            f"Significance table skipped: at most {sample_sizes[-1]} matched pair(s) per comparison over pairing key "
            f"{pairing_cols}, which is not enough for any paired statistic. Sweep over e.g. the sample index so that "
            f"each method is evaluated on several shared units."
        )
        return None
    if sample_sizes[0] < 10:
        logging.warning(
            f"Significance table: only {n_str} paired sample(s) per comparison. Both the {ci_method} CI and the "
            f"Wilcoxon p-values are unreliable at this sample size (a two-sided Wilcoxon test cannot go below "
            f"p=0.125 with n=4); widen the run sweep to get meaningful significance statements."
        )

    title = (
        f"Paired comparison vs. {_label_for(primary, method_labels)} "
        f"(n={n_str} over {', '.join(pairing_cols)}, {ci_level:.0%} {ci_method} CI, Wilcoxon {alternative})"
    )
    return _sanitize_table_df(table_df), title

"""
This task plots the results of a multistage sweep, where each stage has its own set of sweep combinations.
So the structure is that we have L stages, where each stage has n_i sweep combinations.

    Parameters:
        - metric: The metric to plot, e.g. "loss", "accuracy", should must be an accessible key in the CacheableDict.
        - run_results: A list of CacheableDict objects,

    We assume that the run_results is a list of CacheableDict, from which we can extract a certain parameter.

    Examples:
        - L=1, n_1=5 (single metric) -> plot ser over number of iterations
        - L=2, n_1=5, n_2=3: -> recon


    Basically, throughout the multiple stages we can consider the final results as a list of results, each having one sweep combo, combined throughout the stages. For L=2 this leads to a sweep combo depth of 2, where we can pick one as x axis and the other via color grading.
    For all other sweep combo depths it is not clear yet how we can represent it (maybe taking averages or something like that). 
"""

@task(name="Seaborn plot", tags=["plot"], version="1.0", retries=0, description="Plot the results of a multistage sweep.")
def seaborn_plot_sweep_results_auto(
        # df_results : pd.DataFrame,
        results : List[CacheableDict], # or rather PrefectFuture
        df_sweep : pd.DataFrame,
        wandb_params_task: WandbParamsTask,
        palette: str,
        figsize_cfg: CacheableListConfig,
        type : str,
        stochastic_params_cfg: CacheableListConfig, #Optional[List[str]] = None,
        tradeoff_plots_enabled: bool = True,
        tradeoff_same_color_lines_enabled: bool = False,
        tradeoff_errorbars_enabled: bool = False,
        paper_summary_table_enabled: bool = False,
        paper_summary_metrics_filter_cfg: Optional[CacheableListConfig] = None,
        paper_winrate_table_enabled: bool = False,
        paper_winrate_metrics_filter_cfg: Optional[CacheableListConfig] = None,
        winrate_over_param: str = "sample_index",
        paper_significance_table_enabled: bool = False,
        paper_significance_primary: Optional[str] = None,
        paper_significance_method_param: str = "+exps",
        paper_significance_metrics_filter_cfg: Optional[CacheableListConfig] = None,
        paper_significance_pairing_params_cfg: Optional[CacheableListConfig] = None,
        paper_significance_baselines_cfg: Optional[CacheableListConfig] = None,
        paper_significance_method_labels_cfg: Optional[CacheableDictConfig] = None,
        paper_significance_metric_labels_cfg: Optional[CacheableDictConfig] = None,
        paper_significance_ci_method: str = "bootstrap",
        paper_significance_ci_level: float = 0.95,
        paper_significance_bootstrap_samples: int = 10000,
        paper_significance_bootstrap_seed: int = 0,
        paper_significance_alternative: str = "two-sided",
        paper_significance_pvalue_correction: str = "none",
        paper_significance_split_param: Optional[str] = None,
        paper_significance_split_labels_cfg: Optional[CacheableDictConfig] = None,
        rollout_step_plot_enabled: bool = False,
        rollout_step_metrics_cfg: Optional[CacheableListConfig] = None,
        rollout_step_hue_param: str = "+exps",
        # Cost-vs-quality plot + table (appendix/13_computational_cost): one curve per method over a compute budget
        # (iterations / NFE), x = cost metric (ReconComputeStats), y = quality metric.
        paper_cost_plot_enabled: bool = False,
        paper_cost_plot_method_param: str = "+exps",
        paper_cost_plot_budget_params_cfg: Optional[CacheableListConfig] = None,
        paper_cost_plot_x_metrics_cfg: Optional[CacheableListConfig] = None,
        paper_cost_plot_y_metrics_cfg: Optional[CacheableListConfig] = None,
        paper_cost_plot_method_labels_cfg: Optional[CacheableDictConfig] = None,
        paper_cost_plot_metric_labels_cfg: Optional[CacheableDictConfig] = None,
        paper_cost_plot_log_x: bool = True,
        log_lin_scale_auto_enabled : bool = True,
        log_lin_scale_threshold : float = 2.0,
        dpi : int = 100
    ) -> None:

    """
        Parameters:
            - df_results: A pandas DataFrame containing the results of the sweep.
            - df_sweep: A pandas DataFrame containing the sweep combinations.
            - wandb_params_task: The WandbParamsTask object containing the parameters for the wandb run.

        Returns:
            None, but logs the plot to wandb.
    """

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    df_results = pd.DataFrame([_result_to_dict(result) for result in results])

    with wandb.init(**wandb_kwargs_for_prefect_task(wandb_params_task)):

        # First we log the dataframes to wandb

        # Identify numeric metrics in results
        numeric_metrics = df_results.select_dtypes(include=['number']).columns.tolist()
        filtered_numeric_metrics = _filter_metrics_by_cfg(
            metrics=numeric_metrics,
            metrics_filter_cfg=paper_summary_metrics_filter_cfg,
            warning_context="summary/winrate/tradeoff",
        )
        print(f"Ignoring the following result params: {df_results.columns.difference(numeric_metrics).tolist()}")

        # Identify sweep parameters (numeric or categorical)
        sweep_params = df_sweep.columns.tolist()

        # aggregate the results and sweep combinations
        df_plot = pd.concat([df_results, df_sweep], axis=1)
        stochastic_params_requested = stochastic_params_cfg.cfg # if stochastic_params is not None else []
        stochastic_params = []
        for param in stochastic_params_requested:
            resolved_param = _resolve_column_name(param, df_plot.columns.tolist())
            if resolved_param is not None:
                stochastic_params.append(resolved_param)
            else:
                logging.warning(f"Stochastic parameter '{param}' not found in dataframe columns; skipping it.")

        df_plot_stats = df_plot
        metric_mean_cols = {}
        metric_std_cols = {}
        all_params_stochastic = False
        non_stochastic_sweep_params = [param for param in sweep_params if param not in stochastic_params]
        if stochastic_params:
            group_cols = [param for param in sweep_params if param not in stochastic_params]
            if group_cols:
                agg = df_plot.groupby(group_cols, dropna=False)[numeric_metrics].agg(["mean", "std"])
                agg.columns = [f"{metric}__{stat}" for metric, stat in agg.columns]
                df_plot_stats = agg.reset_index()
                metric_mean_cols = {metric: f"{metric}__mean" for metric in numeric_metrics}
                metric_std_cols = {metric: f"{metric}__std" for metric in numeric_metrics}
            else:
                logging.warning("All sweep params are marked stochastic; skipping aggregation.")
                all_params_stochastic = True
        wandb.log({"dataframe": wandb.Table(dataframe=df_plot)})

        if paper_summary_table_enabled:
            summary_numeric_metrics = filtered_numeric_metrics

            summary_table_df = _build_summary_table(
                df_plot=df_plot,
                sweep_params=sweep_params,
                stochastic_params=stochastic_params,
                numeric_metrics=summary_numeric_metrics,
            )
            wandb.log({"paper_summary_table/data": wandb.Table(dataframe=summary_table_df)})
            summary_fig = _render_table_figure(
                table_df=summary_table_df,
                title="Sweep Summary (mean ± std)",
                figsize=figsize_cfg.cfg,
                dpi=dpi,
            )
            wandb.log({"paper_summary_table/figure": wandb.Image(summary_fig)})
            plt.close(summary_fig)

        if paper_winrate_table_enabled:
            # Separate filter so the summary table can carry descriptive, non-comparable
            # columns (e.g. the flow-degradation metrics, which are a property of the run's
            # input rather than of its output) without them turning into meaningless
            # "win-rate" columns. Defaults to the summary filter when not set.
            winrate_filter_is_set = paper_winrate_metrics_filter_cfg is not None and any(
                str(v).strip() for v in paper_winrate_metrics_filter_cfg.cfg
            )
            winrate_numeric_metrics = (
                _filter_metrics_by_cfg(
                    metrics=numeric_metrics,
                    metrics_filter_cfg=paper_winrate_metrics_filter_cfg,
                    warning_context="winrate",
                )
                if winrate_filter_is_set
                else filtered_numeric_metrics
            )

            effective_winrate_over_param = _resolve_column_name(winrate_over_param, df_plot.columns.tolist())
            if effective_winrate_over_param is None:
                if len(stochastic_params) > 0:
                    effective_winrate_over_param = stochastic_params[0]
                    logging.warning(
                        f"Requested winrate_over_param='{winrate_over_param}' not found; "
                        f"falling back to stochastic parameter '{effective_winrate_over_param}'."
                    )
                else:
                    logging.warning(
                        "Win-rate table skipped: requested winrate_over_param is missing and no stochastic "
                        "parameters are available in the sweep dataframe."
                    )

            winrate_table_df = None
            if effective_winrate_over_param is not None:
                winrate_table_df = _build_winrate_table(
                    df_plot=df_plot,
                    sweep_params=sweep_params,
                    stochastic_params=stochastic_params,
                    numeric_metrics=winrate_numeric_metrics,
                    winrate_over_param=effective_winrate_over_param,
                )
            if winrate_table_df is not None and not winrate_table_df.empty:
                wandb.log({"paper_winrate_table/data": wandb.Table(dataframe=winrate_table_df)})
                winrate_fig = _render_table_figure(
                    table_df=winrate_table_df,
                    title=f"Win-rate over {effective_winrate_over_param}",
                    figsize=figsize_cfg.cfg,
                    dpi=dpi,
                )
                wandb.log({"paper_winrate_table/figure": wandb.Image(winrate_fig)})
                plt.close(winrate_fig)
            else:
                logging.warning("Win-rate table requested but could not be constructed with current inputs.")

        if paper_significance_table_enabled:
            if paper_significance_primary is None or not str(paper_significance_primary).strip():
                logging.warning(
                    "Significance table skipped: paper_significance_primary is not set, so there is no primary "
                    "method to compare the other configurations against."
                )
            else:
                # Same convention as the win-rate table: an unset dedicated filter reuses the
                # summary filter, so a flow only has to list its metrics once.
                significance_filter_is_set = paper_significance_metrics_filter_cfg is not None and any(
                    str(v).strip() for v in paper_significance_metrics_filter_cfg.cfg
                )
                active_significance_filter_cfg = (
                    paper_significance_metrics_filter_cfg if significance_filter_is_set else paper_summary_metrics_filter_cfg
                )
                significance_numeric_metrics = _order_metrics_by_filter(
                    metrics=(
                        _filter_metrics_by_cfg(
                            metrics=numeric_metrics,
                            metrics_filter_cfg=paper_significance_metrics_filter_cfg,
                            warning_context="significance",
                        )
                        if significance_filter_is_set
                        else filtered_numeric_metrics
                    ),
                    metrics_filter_cfg=active_significance_filter_cfg,
                )

                # Optionally one table per value of a sweep parameter (e.g. one per
                # domain_specific_setup when a flow sweeps several tasks at once): pairing over
                # the sample index alone would otherwise silently average the tasks together.
                significance_splits: List[Tuple[Optional[Any], pd.DataFrame]] = [(None, df_plot)]
                split_col = None
                if paper_significance_split_param is not None and str(paper_significance_split_param).strip():
                    split_col = _resolve_column_name(str(paper_significance_split_param), df_plot.columns.tolist())
                    if split_col is None:
                        logging.warning(
                            f"Significance table: split parameter '{paper_significance_split_param}' is not a column "
                            f"of the sweep dataframe (available: {df_plot.columns.tolist()}); building a single table."
                        )
                    else:
                        split_values = [v for v in dict.fromkeys(df_plot[split_col].tolist()) if not pd.isna(v)]
                        significance_splits = [
                            (split_value, df_plot[df_plot[split_col].astype(str) == str(split_value)])
                            for split_value in split_values
                        ]
                split_labels = _config_dict_to_str_map(paper_significance_split_labels_cfg)

                for split_value, df_split in significance_splits:
                    significance_result = _build_significance_table(
                        df_plot=df_split,
                        sweep_params=sweep_params,
                        numeric_metrics=significance_numeric_metrics,
                        method_param=paper_significance_method_param,
                        primary_value=str(paper_significance_primary),
                        requested_pairing_params=_config_list_to_str_list(paper_significance_pairing_params_cfg),
                        requested_baselines=_config_list_to_str_list(paper_significance_baselines_cfg),
                        method_labels=_config_dict_to_str_map(paper_significance_method_labels_cfg),
                        metric_labels=_config_dict_to_str_map(paper_significance_metric_labels_cfg),
                        ci_method=str(paper_significance_ci_method).lower(),
                        ci_level=float(paper_significance_ci_level),
                        bootstrap_samples=int(paper_significance_bootstrap_samples),
                        bootstrap_seed=int(paper_significance_bootstrap_seed),
                        alternative=str(paper_significance_alternative),
                        pvalue_correction=str(paper_significance_pvalue_correction),
                    )
                    if split_value is None:
                        log_prefix = "paper_significance_table"
                        title_prefix = ""
                    else:
                        split_label = split_labels.get(str(split_value), str(split_value))
                        log_prefix = f"paper_significance_table/{re.sub(r'[^A-Za-z0-9_.-]+', '_', str(split_value))}"
                        title_prefix = f"[{split_label}] "
                    if significance_result is None:
                        logging.warning(
                            f"Significance table requested but could not be constructed with current inputs"
                            + (f" (split {split_col}={split_value})." if split_value is not None else ".")
                        )
                        continue
                    significance_table_df, significance_title = significance_result
                    significance_title = title_prefix + significance_title
                    print(f"{significance_title}\n{significance_table_df.to_string(index=False)}")
                    print(significance_table_df.to_latex(index=False))
                    wandb.log({f"{log_prefix}/data": wandb.Table(dataframe=significance_table_df)})
                    significance_fig = _render_table_figure(
                        table_df=significance_table_df,
                        title=significance_title,
                        figsize=figsize_cfg.cfg,
                        dpi=dpi,
                        auto_column_width=True,
                    )
                    wandb.log({f"{log_prefix}/figure": wandb.Image(significance_fig)})
                    plt.close(significance_fig)

        if rollout_step_plot_enabled:
            rollout_step_metrics = _detect_rollout_step_metrics(df_results.columns.tolist(), rollout_step_metrics_cfg)
            if not rollout_step_metrics:
                logging.warning(
                    "Rollout-step plot requested but no '<metric>_frame_<k>' columns were found in the results "
                    "(enable per_frame_metrics on the domain_specific_setup to produce them)."
                )
            else:
                rollout_hue_col = _resolve_column_name(rollout_step_hue_param, df_plot.columns.tolist())
                for base_metric, step_to_col in rollout_step_metrics.items():
                    steps = sorted(step_to_col.keys())
                    id_cols = [rollout_hue_col] if rollout_hue_col else []
                    long_df = df_plot.melt(
                        id_vars=id_cols,
                        value_vars=[step_to_col[step] for step in steps],
                        var_name="_rollout_step_col",
                        value_name=base_metric,
                    )
                    col_to_step = {col: step for step, col in step_to_col.items()}
                    long_df["rollout_step"] = long_df["_rollout_step_col"].map(col_to_step)

                    plt.figure(figsize=figsize_cfg.cfg, dpi=dpi)
                    sns.lineplot(
                        data=long_df,
                        x="rollout_step",
                        y=base_metric,
                        hue=rollout_hue_col,
                        palette=palette if rollout_hue_col else None,
                        marker="o",
                        errorbar="sd",
                    )
                    if rollout_hue_col:
                        plt.legend(title=rollout_hue_col)
                    plt.title(f"{base_metric} vs rollout step (mean ± std over sweep)")
                    plt.xlabel("rollout step (frames ahead of anchor)")
                    plt.ylabel(base_metric)
                    plt.xticks(steps)
                    plt.tight_layout()
                    wandb.log({f"rollout_step__{base_metric}": wandb.Image(plt)})
                    plt.close()

        if paper_cost_plot_enabled:
            available_cols = df_plot.columns.tolist()
            cost_method_col = _resolve_column_name(paper_cost_plot_method_param, available_cols)
            budget_params = [str(p) for p in paper_cost_plot_budget_params_cfg.cfg] if paper_cost_plot_budget_params_cfg is not None else []
            budget_cols = [col for col in (_resolve_column_name(p, available_cols) for p in budget_params) if col is not None]
            x_requested = [str(m) for m in paper_cost_plot_x_metrics_cfg.cfg] if paper_cost_plot_x_metrics_cfg is not None else ["recon_runtime_s", "recon_peak_gpu_memory_gb"]
            y_requested = [str(m) for m in paper_cost_plot_y_metrics_cfg.cfg] if paper_cost_plot_y_metrics_cfg is not None else ["psnr"]
            cost_x_metrics = [m for m in x_requested if m in numeric_metrics]
            cost_y_metrics = [m for m in y_requested if m in numeric_metrics]
            if cost_method_col is None or not budget_cols or not cost_x_metrics or not cost_y_metrics:
                logging.warning(
                    f"Cost plot requested but inputs are incomplete: method column {paper_cost_plot_method_param!r} -> {cost_method_col}, "
                    f"budget params {budget_params} -> {budget_cols}, x metrics {x_requested} -> {cost_x_metrics}, "
                    f"y metrics {y_requested} -> {cost_y_metrics} (numeric metrics: {numeric_metrics})."
                )
            else:
                cost_method_labels = dict(OmegaConf.to_container(paper_cost_plot_method_labels_cfg.cfg)) if paper_cost_plot_method_labels_cfg is not None else {}
                cost_metric_labels = dict(OmegaConf.to_container(paper_cost_plot_metric_labels_cfg.cfg)) if paper_cost_plot_metric_labels_cfg is not None else {}
                cost_metrics = cost_y_metrics + [m for m in cost_x_metrics if m not in cost_y_metrics]
                cost_df = _build_cost_quality_table(df_plot, cost_method_col, budget_cols, cost_metrics)
                print(f"Cost vs quality (per method and budget)\n{cost_df.to_string(index=False)}")
                wandb.log({"paper_cost_table/data": wandb.Table(dataframe=cost_df)})

                cost_table_df = pd.DataFrame({
                    "Method": [cost_method_labels.get(m, m) for m in cost_df[cost_method_col]],
                    "Budget": cost_df["budget"],
                    **{
                        cost_metric_labels.get(metric, metric): [
                            f"{mean:.4g} ± {0.0 if pd.isna(std) else std:.2g}"
                            for mean, std in zip(cost_df[f"{metric}__mean"], cost_df[f"{metric}__std"])
                        ]
                        for metric in cost_metrics
                    },
                    "n": cost_df["n"],
                })
                print(cost_table_df.to_latex(index=False))
                cost_table_fig = _render_table_figure(
                    table_df=cost_table_df,
                    title="Cost vs quality (budget = iterations / NFE)",
                    figsize=figsize_cfg.cfg,
                    dpi=dpi,
                    auto_column_width=True,
                )
                wandb.log({"paper_cost_table/figure": wandb.Image(cost_table_fig)})
                plt.close(cost_table_fig)

                cost_methods = list(dict.fromkeys(cost_df[cost_method_col].tolist()))
                cost_colors = dict(zip(cost_methods, sns.color_palette(palette, n_colors=max(1, len(cost_methods)))))
                for x_metric in cost_x_metrics:
                    for y_metric in cost_y_metrics:
                        fig, ax = plt.subplots(figsize=figsize_cfg.cfg, dpi=dpi)
                        for method in cost_methods:
                            method_df = cost_df[cost_df[cost_method_col] == method].sort_values("budget")
                            x_mean, y_mean = method_df[f"{x_metric}__mean"], method_df[f"{y_metric}__mean"]
                            ax.errorbar(
                                x_mean, y_mean,
                                xerr=method_df[f"{x_metric}__std"].fillna(0.0),
                                yerr=method_df[f"{y_metric}__std"].fillna(0.0),
                                marker="o", capsize=3, color=cost_colors[method],
                                label=cost_method_labels.get(method, method),
                            )
                            for budget, x_val, y_val in zip(method_df["budget"], x_mean, y_mean):
                                ax.annotate(str(budget), (x_val, y_val), textcoords="offset points", xytext=(4, 4), fontsize=7, color=cost_colors[method])
                        if paper_cost_plot_log_x:
                            ax.set_xscale("log")
                        ax.set_xlabel(cost_metric_labels.get(x_metric, x_metric))
                        ax.set_ylabel(cost_metric_labels.get(y_metric, y_metric))
                        ax.set_title(f"{cost_metric_labels.get(y_metric, y_metric)} vs {cost_metric_labels.get(x_metric, x_metric)} (point labels: iterations / NFE)")
                        ax.grid(alpha=0.3)
                        ax.legend()
                        fig.tight_layout()
                        wandb.log({f"paper_cost_plot/{y_metric}__vs__{x_metric}": wandb.Image(fig)})
                        plt.close(fig)

        plot_fn = sns.lineplot if type == "line" else sns.scatterplot

        # For each metric, plot against each sweep parameter (x), and optionally color by a different sweep param
        for metric in numeric_metrics:
            if all_params_stochastic:
                plt.figure(figsize=figsize_cfg.cfg, dpi=dpi)
                df_plot_all = df_plot.assign(_all="all")
                sns.violinplot(
                    data=df_plot_all,
                    x="_all",
                    y=metric,
                    palette=palette,
                    inner="quartile",
                    cut=0,
                )
                sns.stripplot(
                    data=df_plot_all,
                    x="_all",
                    y=metric,
                    color="black",
                    alpha=0.4,
                    jitter=True,
                    size=2,
                )
                plot_title = f"{metric} distribution (stochastic sweep)"
                plt.title(plot_title)
                plt.xlabel("all")
                plt.ylabel(metric)
                plt.xticks(rotation=45)
                if log_lin_scale_auto_enabled:
                    y_values = df_plot[metric]
                    y_ratio = y_values.max() / y_values.min()
                    plt.yscale('log' if y_ratio > math.pow(10, log_lin_scale_threshold) else 'linear')
                plt.tight_layout()
                wandb.log({f"{metric}__distribution": wandb.Image(plt)})
                plt.close()
                continue
            if len(non_stochastic_sweep_params) == 1 and len(sweep_params) > 1:
                sweep_param = non_stochastic_sweep_params[0]
                plt.figure(figsize=figsize_cfg.cfg, dpi=dpi)
                sns.violinplot(
                    data=df_plot,
                    x=sweep_param,
                    y=metric,
                    palette=palette,
                    inner="quartile",
                    cut=0,
                )
                sns.stripplot(
                    data=df_plot,
                    x=sweep_param,
                    y=metric,
                    color="black",
                    alpha=0.4,
                    jitter=True,
                    size=2,
                )
                plot_title = f"{metric} distribution vs {sweep_param}"
                plt.title(plot_title)
                plt.xlabel(sweep_param)
                plt.ylabel(metric)
                plt.xticks(rotation=45)
                if log_lin_scale_auto_enabled:
                    y_values = df_plot[metric]
                    y_ratio = y_values.max() / y_values.min()
                    plt.yscale('log' if y_ratio > math.pow(10, log_lin_scale_threshold) else 'linear')
                plt.tight_layout()
                wandb.log({f"{metric}__vs__{sweep_param}": wandb.Image(plt)})
                plt.close()
                continue
            for i, x_param in enumerate(sweep_params):
            # Optionally, try coloring by a different sweep param (if available)
                metric_x_param_combination_plotted = False
                for j, hue_param in enumerate(sweep_params):
                    if hue_param == x_param:
                        continue  # Don't use same param for x and hue

                    if stochastic_params and (x_param in stochastic_params or hue_param in stochastic_params):
                        continue

                    # continue if x_param is only one unique value
                    if df_plot_stats[x_param].nunique() <= 1:
                        continue

                    num_of_sweep_values_for_hue_param = df_sweep[hue_param].nunique()
                    if num_of_sweep_values_for_hue_param <= 1:
                        if metric_x_param_combination_plotted:
                            continue
                        else:
                            metric_x_param_combination_plotted = True

                    plt.figure(figsize=figsize_cfg.cfg, dpi=dpi)  # Set desired dpi here
                    y_col = metric_mean_cols.get(metric, metric)
                    std_col = metric_std_cols.get(metric)
                    use_hue = hue_param if len(sweep_params) > 1 and num_of_sweep_values_for_hue_param > 1 else None

                    if type == "line":
                        if use_hue:
                            for hue_value in sorted(df_plot_stats[use_hue].unique()):
                                df_sub = df_plot_stats[df_plot_stats[use_hue] == hue_value].sort_values(x_param)
                                plt.plot(df_sub[x_param], df_sub[y_col], label=str(hue_value))
                                if std_col is not None and pd.api.types.is_numeric_dtype(df_sub[x_param]):
                                    plt.fill_between(
                                        df_sub[x_param],
                                        df_sub[y_col] - df_sub[std_col],
                                        df_sub[y_col] + df_sub[std_col],
                                        alpha=0.2,
                                    )
                            plt.legend(title=use_hue)
                        else:
                            df_sub = df_plot_stats.sort_values(x_param)
                            plt.plot(df_sub[x_param], df_sub[y_col])
                            if std_col is not None and pd.api.types.is_numeric_dtype(df_sub[x_param]):
                                plt.fill_between(
                                    df_sub[x_param],
                                    df_sub[y_col] - df_sub[std_col],
                                    df_sub[y_col] + df_sub[std_col],
                                    alpha=0.2,
                                )
                    else:
                        df_sub = df_plot_stats
                        if std_col is not None:
                            plt.errorbar(
                                df_sub[x_param],
                                df_sub[y_col],
                                yerr=df_sub[std_col],
                                fmt="o",
                                label=None,
                            )
                        else:
                            plot_fn(
                                data=df_plot_stats,
                                x=x_param,
                                y=y_col,
                                hue=use_hue,
                                palette=palette,
                            )
                            if use_hue:
                                plt.legend(title=use_hue)
                    plot_title = f"{metric} vs {x_param}" + (f" cb {hue_param}" if len(sweep_params) > 1 else "")
                    plt.title(plot_title)
                    plt.xlabel(x_param)
                    plt.ylabel(metric)
                    if log_lin_scale_auto_enabled:
                        # first check if the df_plot[x_param] is a numeric type
                        if pd.api.types.is_numeric_dtype(df_plot_stats[x_param]):
                            x_min = df_plot_stats[x_param].min()
                            x_max = df_plot_stats[x_param].max()
                            if x_min > 0 and x_max > 0:
                                x_ratio = x_max / x_min
                                plt.xscale('log' if x_ratio > math.pow(10, log_lin_scale_threshold) else 'linear')
                        y_values = df_plot_stats[y_col]
                        y_min = y_values.min()
                        y_max = y_values.max()
                        if y_min > 0 and y_max > 0:
                            y_ratio = y_max / y_min
                            plt.yscale('log' if y_ratio > math.pow(10, log_lin_scale_threshold) else 'linear')
                    plt.tight_layout()
                    wandb.log({f"{metric}__vs__{x_param}": wandb.Image(plt)})
                    
                # If only one sweep param, plot without hue
                if len(sweep_params) == 1:
                    if stochastic_params and x_param in stochastic_params:
                        continue
                    is_bool_like = (
                        pd.api.types.is_bool_dtype(df_plot[x_param])
                        or set(df_plot[x_param].dropna().unique()).issubset({True, False})
                    )
                    if is_bool_like:
                        plt.figure(figsize=figsize_cfg.cfg, dpi=dpi)
                        sns.violinplot(
                            data=df_plot,
                            x=x_param,
                            y=metric,
                            palette=palette,
                            inner="quartile",
                            cut=0,
                        )
                        sns.stripplot(
                            data=df_plot,
                            x=x_param,
                            y=metric,
                            color="black",
                            alpha=0.4,
                            jitter=True,
                            size=2,
                        )
                        plot_title = f"{metric} distribution vs {x_param}"
                        plt.title(plot_title)
                        plt.xlabel(x_param)
                        plt.ylabel(metric)
                        plt.xticks(rotation=45)
                        if log_lin_scale_auto_enabled:
                            y_values = df_plot[metric]
                            y_ratio = y_values.max() / y_values.min()
                            plt.yscale('log' if y_ratio > math.pow(10, log_lin_scale_threshold) else 'linear')
                        plt.tight_layout()
                        wandb.log({f"{metric}__vs__{x_param}": wandb.Image(plt)})
                        plt.close()
                        continue
                    if df_plot_stats[x_param].nunique() <= 1:
                        plt.figure(figsize=figsize_cfg.cfg, dpi=dpi)
                        sns.violinplot(
                            data=df_plot,
                            x=x_param,
                            y=metric,
                            palette=palette,
                            inner="quartile",
                            cut=0
                        )
                        sns.stripplot(
                            data=df_plot,
                            x=x_param,
                            y=metric,
                            color="black",
                            alpha=0.4,
                            jitter=True,
                            size=2,
                        )
                        plot_title = f"{metric} distribution ({x_param})"
                        plt.title(plot_title)
                        plt.xlabel(x_param)
                        plt.ylabel(metric)
                        plt.xticks(rotation=45)
                        if log_lin_scale_auto_enabled:
                            y_values = df_plot[metric]
                            y_ratio = y_values.max() / y_values.min()
                            plt.yscale('log' if y_ratio > math.pow(10, log_lin_scale_threshold) else 'linear')
                        plt.tight_layout()
                        wandb.log({f"{metric}__vs__{x_param}": wandb.Image(plt)})
                        plt.close()
                        continue

                    plt.figure(figsize=figsize_cfg.cfg, dpi=dpi)
                    y_col = metric_mean_cols.get(metric, metric)
                    std_col = metric_std_cols.get(metric)
                    if type == "line":
                        df_sub = df_plot_stats.sort_values(x_param)
                        plt.plot(df_sub[x_param], df_sub[y_col])
                        if std_col is not None and pd.api.types.is_numeric_dtype(df_sub[x_param]):
                            plt.fill_between(
                                df_sub[x_param],
                                df_sub[y_col] - df_sub[std_col],
                                df_sub[y_col] + df_sub[std_col],
                                alpha=0.2,
                            )
                    else:
                        if std_col is not None:
                            plt.errorbar(
                                df_plot_stats[x_param],
                                df_plot_stats[y_col],
                                yerr=df_plot_stats[std_col],
                                fmt="o",
                            )
                        else:
                            plot_fn(
                                data=df_plot_stats,
                                x=x_param,
                                y=y_col,
                                palette=palette,
                            )
                    plot_title = f"{metric} vs {x_param}"
                    plt.title(plot_title)
                    plt.xlabel(x_param)
                    plt.ylabel(metric)
                    if log_lin_scale_auto_enabled:
                        if pd.api.types.is_numeric_dtype(df_plot_stats[x_param]):
                            x_min = df_plot_stats[x_param].min()
                            x_max = df_plot_stats[x_param].max()
                            if x_min > 0 and x_max > 0:
                                x_ratio = x_max / x_min
                                plt.xscale('log' if x_ratio > math.pow(10, log_lin_scale_threshold) else 'linear')
                        y_values = df_plot_stats[y_col]
                        y_min = y_values.min()
                        y_max = y_values.max()
                        if y_min > 0 and y_max > 0:
                            y_ratio = y_max / y_min
                            plt.yscale('log' if y_ratio > math.pow(10, log_lin_scale_threshold) else 'linear')
                    plt.tight_layout()
                    wandb.log({f"{metric}__vs__{x_param}": wandb.Image(plt)})
                    plt.close()

        if tradeoff_plots_enabled and len(filtered_numeric_metrics) >= 2:
            tradeoff_group_params = [param for param in sweep_params if param not in stochastic_params]
            if stochastic_params:
                if tradeoff_group_params:
                    tradeoff_agg = df_plot.groupby(tradeoff_group_params, dropna=False)[filtered_numeric_metrics].agg(["mean", "std"])
                    tradeoff_agg.columns = [f"{metric}__{stat}" for metric, stat in tradeoff_agg.columns]
                    tradeoff_df = tradeoff_agg.reset_index()
                else:
                    tradeoff_stats = {f"{metric}__mean": [df_plot[metric].mean()] for metric in filtered_numeric_metrics}
                    tradeoff_stats.update({f"{metric}__std": [df_plot[metric].std()] for metric in filtered_numeric_metrics})
                    tradeoff_df = pd.DataFrame(tradeoff_stats)
                    tradeoff_df["_all"] = "all"
                    tradeoff_group_params = ["_all"]
            else:
                tradeoff_df = df_plot.copy()
                tradeoff_group_params = sweep_params.copy()

            tradeoff_df = tradeoff_df.copy()
            color_param = tradeoff_group_params[0] if len(tradeoff_group_params) >= 1 else None
            style_param = tradeoff_group_params[1] if len(tradeoff_group_params) >= 2 else None
            remaining_params = tradeoff_group_params[2:] if len(tradeoff_group_params) > 2 else []

            if color_param is not None:
                color_values = list(dict.fromkeys(tradeoff_df[color_param].tolist()))
            else:
                color_values = ["all"]
                tradeoff_df["_color_val"] = "all"
            if style_param is not None:
                style_values = list(dict.fromkeys(tradeoff_df[style_param].tolist()))
            else:
                style_values = ["all"]
                tradeoff_df["_style_val"] = "all"

            if color_param is not None:
                tradeoff_df["_color_val"] = tradeoff_df[color_param]
            if style_param is not None:
                tradeoff_df["_style_val"] = tradeoff_df[style_param]

            color_palette_vals = sns.color_palette(palette, n_colors=max(1, len(color_values)))
            color_to_rgb = {val: color_palette_vals[idx] for idx, val in enumerate(color_values)}

            line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
            style_to_linestyle = {val: line_styles[idx % len(line_styles)] for idx, val in enumerate(style_values)}

            if remaining_params:
                tradeoff_df["_series_key"] = tradeoff_df[remaining_params].astype(str).agg(" | ".join, axis=1)
            else:
                tradeoff_df["_series_key"] = "all"

            def _series_sort(df_series: pd.DataFrame) -> pd.DataFrame:
                if color_param is not None and pd.api.types.is_numeric_dtype(df_series[color_param]):
                    return df_series.sort_values(color_param)
                return df_series.sort_values("_x_plot")

            for x_metric, y_metric in permutations(filtered_numeric_metrics, 2):
                x_col = f"{x_metric}__mean" if stochastic_params else x_metric
                y_col = f"{y_metric}__mean" if stochastic_params else y_metric
                x_std_col = f"{x_metric}__std" if stochastic_params else None
                y_std_col = f"{y_metric}__std" if stochastic_params else None

                tradeoff_df["_x_plot"] = tradeoff_df[x_col]
                tradeoff_df["_y_plot"] = tradeoff_df[y_col]

                plt.figure(figsize=figsize_cfg.cfg, dpi=dpi)

                # Isolines: fix style and remaining params, vary color param if available
                if color_param is not None and style_param is not None:
                    for (style_val, series_key), iso_df in tradeoff_df.groupby(["_style_val", "_series_key"], dropna=False):
                        iso_df = _series_sort(iso_df)
                        if len(iso_df) > 1:
                            plt.plot(
                                iso_df["_x_plot"],
                                iso_df["_y_plot"],
                                linestyle=style_to_linestyle[style_val],
                                color="gray",
                                alpha=0.30,
                                linewidth=1.0,
                                zorder=1,
                            )

                    if tradeoff_same_color_lines_enabled:
                        # optional cross-isolines: fix color and remaining params, vary style param
                        for (color_val, series_key), iso_df in tradeoff_df.groupby(["_color_val", "_series_key"], dropna=False):
                            if style_param is not None and pd.api.types.is_numeric_dtype(iso_df[style_param]):
                                iso_df = iso_df.sort_values(style_param)
                            else:
                                iso_df = iso_df.sort_values("_x_plot")
                            if len(iso_df) > 1:
                                plt.plot(
                                    iso_df["_x_plot"],
                                    iso_df["_y_plot"],
                                    linestyle="-",
                                    color=color_to_rgb[color_val],
                                    alpha=0.22,
                                    linewidth=0.9,
                                    zorder=1,
                                )

                for _, row in tradeoff_df.iterrows():
                    point_color = color_to_rgb[row["_color_val"]]
                    point_linestyle = style_to_linestyle[row["_style_val"]]
                    xerr = row[x_std_col] if x_std_col and x_std_col in row.index else None
                    yerr = row[y_std_col] if y_std_col and y_std_col in row.index else None
                    if xerr is not None and pd.isna(xerr):
                        xerr = None
                    if yerr is not None and pd.isna(yerr):
                        yerr = None

                    if stochastic_params and tradeoff_errorbars_enabled:
                        plt.errorbar(
                            x=row[x_col],
                            y=row[y_col],
                            xerr=xerr,
                            yerr=yerr,
                            fmt="o",
                            color=point_color,
                            alpha=0.85,
                            capsize=3,
                            linestyle=point_linestyle,
                        )
                    else:
                        plt.scatter(
                            row[x_col],
                            row[y_col],
                            color=point_color,
                            marker="o",
                            alpha=0.85,
                            zorder=3,
                        )

                color_handles = [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=color_to_rgb[val],
                        markeredgecolor=color_to_rgb[val],
                        markersize=6,
                        linestyle="",
                        label=str(val),
                    )
                    for val in color_values
                ]
                style_handles = [
                    Line2D(
                        [0],
                        [0],
                        color="black",
                        linestyle=style_to_linestyle[val],
                        linewidth=1.5,
                        label=str(val),
                    )
                    for val in style_values
                ]

                legend_anchor_y = -0.12
                if len(color_handles) > 1 or color_param is not None:
                    color_legend = plt.legend(
                        handles=color_handles,
                        title=color_param if color_param is not None else "color",
                        loc="upper center",
                        bbox_to_anchor=(0.5, legend_anchor_y),
                        borderaxespad=0.0,
                        ncol=min(6, max(1, len(color_handles))),
                        frameon=False,
                    )
                    plt.gca().add_artist(color_legend)
                    legend_anchor_y -= 0.14

                if len(style_handles) > 1 or style_param is not None:
                    plt.legend(
                        handles=style_handles,
                        title=style_param if style_param is not None else "style",
                        loc="upper center",
                        bbox_to_anchor=(0.5, legend_anchor_y),
                        borderaxespad=0.0,
                        ncol=min(6, max(1, len(style_handles))),
                        frameon=False,
                    )

                plt.title(f"Tradeoff: {y_metric} vs {x_metric}")
                plt.xlabel(x_metric)
                plt.ylabel(y_metric)

                if log_lin_scale_auto_enabled:
                    x_values = tradeoff_df[x_col]
                    y_values = tradeoff_df[y_col]
                    x_min = x_values.min()
                    x_max = x_values.max()
                    y_min = y_values.min()
                    y_max = y_values.max()
                    if x_min > 0 and x_max > 0:
                        x_ratio = x_max / x_min
                        plt.xscale('log' if x_ratio > math.pow(10, log_lin_scale_threshold) else 'linear')
                    if y_min > 0 and y_max > 0:
                        y_ratio = y_max / y_min
                        plt.yscale('log' if y_ratio > math.pow(10, log_lin_scale_threshold) else 'linear')

                plt.tight_layout()
                if style_param is not None:
                    plt.subplots_adjust(bottom=0.30)
                elif color_param is not None:
                    plt.subplots_adjust(bottom=0.20)
                wandb.log({f"tradeoff__{y_metric}__vs__{x_metric}": wandb.Image(plt)})
                plt.close()