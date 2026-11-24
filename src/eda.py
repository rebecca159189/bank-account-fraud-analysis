"""Full-population descriptive analysis of the BAF Base data.

This module does not train predictive models. Only the training months (0--5)
define the reference bins used for the descriptive PSI tables. The parquet cache
preserves the source values, including documented missing-value sentinels.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MINUS_ONE_MISSING = (
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
)
NEGATIVE_MISSING = ("intended_balcon_amount",)
CATEGORICAL = ("payment_type", "employment_status", "housing_status", "source", "device_os")
KEY_NUMERIC = (
    "income",
    "name_email_similarity",
    "customer_age",
    "velocity_6h",
    "velocity_24h",
    "credit_risk_score",
    "intended_balcon_amount",
    "session_length_in_minutes",
)
SPLITS = {"train": (0, 1, 2, 3, 4, 5), "validation": (6,), "test": (7,)}
PSI_LOW_CARDINALITY_MAX = 20


def documented_missing_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    """Apply only rules stated in references/datasheet.txt, Q7."""
    result = frame[column].isna()
    if column in MINUS_ONE_MISSING:
        result |= frame[column].eq(-1)
    elif column in NEGATIVE_MISSING:
        result |= frame[column].lt(0)
    return result


def clean_for_description(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in MINUS_ONE_MISSING + NEGATIVE_MISSING:
        result.loc[documented_missing_mask(frame, column), column] = np.nan
    return result


def split_for_month(month: int) -> str:
    return next(name for name, months in SPLITS.items() if month in months)


def rate_stats(frame: pd.DataFrame) -> dict:
    n = int(len(frame))
    fraud = int(frame["fraud_bool"].sum())
    return {"rows": n, "fraud_count": fraud, "legitimate_count": n - fraud,
            "fraud_rate": fraud / n if n else np.nan}


def fit_psi_bins(reference: pd.Series, categorical: bool = False) -> dict:
    """Define bins using the reference only, including explicit missing bins.

    Categorical and numeric fields with at most 20 observed reference values
    receive one bin per reference value, a reserved unseen-value bin, and a
    missing bin. This preserves binary shifts even when decile edges would
    collapse to only 0 and 1. Remaining numeric fields use reference deciles.
    """
    observed = reference.dropna()
    if categorical or observed.nunique() <= PSI_LOW_CARDINALITY_MAX:
        return {"kind": "categorical" if categorical else "discrete",
                "levels": observed.drop_duplicates().tolist(),
                "unseen_bin": True, "missing_bin": True}
    values = observed.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("PSI reference contains non-finite numeric values other than missing values.")
    quantiles = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
    return {"kind": "continuous", "edges": np.r_[-np.inf, quantiles[1:-1], np.inf].tolist(),
            "missing_bin": True}


def _psi_bin_counts(series: pd.Series, specification: dict) -> np.ndarray:
    if specification["kind"] in {"categorical", "discrete"}:
        levels = specification["levels"]
        counts = series.value_counts(dropna=True).reindex(pd.Index(levels), fill_value=0).to_numpy(float)
        unseen = int((series.notna() & ~series.isin(levels)).sum())
        return np.r_[counts, unseen, int(series.isna().sum())]
    return np.r_[np.histogram(series.dropna().to_numpy(dtype=float), bins=specification["edges"])[0],
                 int(series.isna().sum())]


def calculate_psi(reference: pd.Series, current: pd.Series, categorical: bool) -> tuple[float, int]:
    """Training-defined bins and 0.5 pseudocount; no current-data bin fitting."""
    specification = fit_psi_bins(reference, categorical)
    expected_counts = _psi_bin_counts(reference, specification)
    actual_counts = _psi_bin_counts(current, specification)
    if expected_counts.sum() != len(reference) or actual_counts.sum() != len(current):
        raise ValueError("PSI binning did not account for every observation.")
    expected_fraction = (expected_counts + 0.5) / (expected_counts.sum() + 0.5 * len(expected_counts))
    actual_fraction = (actual_counts + 0.5) / (actual_counts.sum() + 0.5 * len(actual_counts))
    value = np.sum((actual_fraction - expected_fraction) * np.log(actual_fraction / expected_fraction))
    return float(value), len(expected_counts)


def set_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120,
        "savefig.dpi": 160, "axes.grid": True, "grid.alpha": 0.2,
    })


def save_plots(tables: dict[str, pd.DataFrame], figure_dir: Path) -> None:
    set_plot_style()
    monthly = tables["monthly"]
    x = monthly["month"].to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    colors = ["#277da8"] * 6 + ["#f8961e", "#7b2cbf"]
    axes[0].bar(x, monthly["rows"], color=colors)
    axes[0].set(ylabel="Applications", title="BAF Base: all 1,000,000 applications over time")
    axes[1].plot(x, monthly["fraud_rate"], color="#d1495b", marker="o", linewidth=2)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for m, p in zip(x, monthly["fraud_rate"]):
        axes[1].annotate(f"{p:.2%}", (m, p), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    axes[1].set(xlabel="Dataset month (0-5 train; 6 validation; 7 test)", ylabel="Fraud prevalence", xticks=x)
    axes[1].margins(y=0.3)
    for ax in axes:
        ax.axvline(5.5, color="#f8961e", linestyle="--", alpha=0.8)
        ax.axvline(6.5, color="#7b2cbf", linestyle="--", alpha=0.8)
    fig.savefig(figure_dir / "eda_monthly_volume_fraud.png")
    plt.close(fig)

    schema = tables["schema"].query("effective_missing_count > 0").sort_values("effective_missing_fraction")
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.barh(schema["column"], schema["effective_missing_fraction"], color="#277da8")
    for i, val in enumerate(schema["effective_missing_fraction"]):
        ax.text(val + 0.005, i, f"{val:.2%}", va="center", fontsize=9)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set(xlabel="Share of all applications", title="Documented missing values (native nulls + source sentinels)")
    ax.set_xlim(0, min(1, max(schema["effective_missing_fraction"], default=0.1) * 1.2))
    fig.savefig(figure_dir / "eda_missing_values.png")
    plt.close(fig)

    age = tables["monthly_age_group"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for group, color in (("age < 50", "#277da8"), ("age >= 50", "#d1495b")):
        subset = age[age["age_group"].eq(group)]
        axes[0].plot(subset["month"], subset["share_within_month"], marker="o", label=group, color=color)
        axes[1].plot(subset["month"], subset["fraud_rate"], marker="o", label=group, color=color)
    for ax in axes:
        ax.set(xlabel="Dataset month", xticks=x)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(frameon=False)
    axes[0].set(ylabel="Share of applications", title="Age-group representation")
    axes[1].set(ylabel="Fraud prevalence", title="Observed label prevalence by age group")
    fig.savefig(figure_dir / "eda_age_group_monthly.png")
    plt.close(fig)

    descriptive = tables["monthly_feature_summary"]
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), constrained_layout=True)
    for ax, feature in zip(axes.flat, KEY_NUMERIC):
        subset = descriptive[descriptive["feature"].eq(feature)]
        ax.fill_between(subset["month"], subset["p10"], subset["p90"], color="#277da8", alpha=0.15, label="10th-90th percentile")
        ax.plot(subset["month"], subset["median"], marker="o", color="#277da8", label="Median")
        ax.set(title=feature, xlabel="Dataset month", xticks=x)
        ax.axvline(5.5, color="#f8961e", linestyle="--", alpha=0.6)
        ax.axvline(6.5, color="#7b2cbf", linestyle="--", alpha=0.6)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Monthly feature distributions after documented sentinel handling", fontsize=15)
    fig.savefig(figure_dir / "eda_monthly_feature_distributions.png")
    plt.close(fig)

    psi = tables["monthly_psi"].pivot(index="feature", columns="month", values="psi")
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    im = ax.imshow(psi.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0)
    ax.grid(False)
    ax.set(xticks=np.arange(len(psi.columns)), xticklabels=psi.columns,
           yticks=np.arange(len(psi.index)), yticklabels=psi.index,
           xlabel="Dataset month", title="Feature PSI relative to pooled training months 0-5")
    for i in range(len(psi.index)):
        for j in range(len(psi.columns)):
            val = psi.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if val > psi.to_numpy().max() * 0.58 else "black")
    fig.colorbar(im, ax=ax, label="PSI (training-defined bins; missing bin; 0.5 pseudocount)")
    fig.savefig(figure_dir / "eda_monthly_psi.png")
    plt.close(fig)


def analyze(input_path: Path, output_dir: Path, cache_path: Path | None, expected_rows: int) -> dict:
    print(f"Reading complete source: {input_path}", flush=True)
    raw = pd.read_csv(input_path)
    if len(raw) != expected_rows:
        raise ValueError(f"Expected {expected_rows:,} rows, found {len(raw):,}; source may be incomplete.")
    if raw["fraud_bool"].isna().any() or not set(raw["fraud_bool"].unique()).issubset({0, 1}):
        raise ValueError("Invalid or missing fraud labels.")
    if set(raw["month"].unique()) != set(range(8)):
        raise ValueError("Expected all dataset month codes 0--7.")
    if raw["customer_age"].isna().any():
        raise ValueError("Age group assignment requires observed customer_age.")
    raw_memory_bytes = int(raw.memory_usage(index=True, deep=True).sum())
    native_missing_count = int(raw.isna().sum().sum())
    duplicate_rows = int(raw.duplicated().sum())
    print(f"Loaded {len(raw):,} rows and {len(raw.columns)} columns; duplicate rows={duplicate_rows}.", flush=True)
    clean = clean_for_description(raw)

    schema_rows = []
    for column in raw.columns:
        numeric = pd.api.types.is_numeric_dtype(raw[column])
        effective_missing = documented_missing_mask(raw, column)
        rule = "native null only"
        if column in MINUS_ONE_MISSING:
            rule = "-1 means missing (official datasheet Q7)"
        elif column in NEGATIVE_MISSING:
            rule = "all negative values mean missing (official datasheet Q7)"
        elif column in {"velocity_6h", "credit_risk_score"}:
            rule = "negative values are valid; retain them (official datasheet Q7 ranges)"
        schema_rows.append({
            "column": column, "raw_dtype": str(raw[column].dtype),
            "role": "target" if column == "fraud_bool" else "split_only" if column == "month" else "feature",
            "native_missing_count": int(raw[column].isna().sum()),
            "minus_one_count": int(raw[column].eq(-1).sum()) if numeric else 0,
            "negative_count": int(raw[column].lt(0).sum()) if numeric else 0,
            "effective_missing_count": int(effective_missing.sum()),
            "effective_missing_fraction": float(effective_missing.mean()),
            "raw_unique_count": int(raw[column].nunique(dropna=False)),
            "constant": bool(raw[column].nunique(dropna=False) <= 1),
            "raw_min": raw[column].min() if numeric else None,
            "raw_max": raw[column].max() if numeric else None,
            "handling": rule,
        })
    tables = {"schema": pd.DataFrame(schema_rows)}
    numeric = clean.select_dtypes(include=np.number)
    tables["numeric_summary"] = numeric.describe(percentiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).T.reset_index(names="column")
    categorical_rows = []
    for column in CATEGORICAL:
        grouped = raw.groupby(column, dropna=False)["fraud_bool"].agg(["size", "sum", "mean"]).reset_index()
        for row in grouped.itertuples(index=False, name=None):
            categorical_rows.append({"feature": column, "category": row[0], "rows": int(row[1]),
                                     "share": row[1] / len(raw), "fraud_count": int(row[2]), "fraud_rate": row[3]})
    tables["category_summary"] = pd.DataFrame(categorical_rows)
    monthly_rows, split_rows, age_rows, monthly_age_rows = [], [], [], []
    for month, group in raw.groupby("month", sort=True):
        monthly_rows.append({"month": int(month), "split": split_for_month(int(month)), **rate_stats(group)})
        for age_group, mask in (("age < 50", group["customer_age"].lt(50)), ("age >= 50", group["customer_age"].ge(50))):
            stats = rate_stats(group.loc[mask])
            monthly_age_rows.append({"month": int(month), "split": split_for_month(int(month)),
                                    "age_group": age_group, **stats, "share_within_month": stats["rows"] / len(group)})
    for name, months in {"all": tuple(range(8)), **SPLITS}.items():
        group = raw.loc[raw["month"].isin(months)]
        split_rows.append({"split": name, "months": ",".join(map(str, months)), **rate_stats(group)})
        for age_group, mask in (("age < 50", group["customer_age"].lt(50)), ("age >= 50", group["customer_age"].ge(50))):
            stats = rate_stats(group.loc[mask])
            age_rows.append({"split": name, "age_group": age_group, **stats, "share_within_split": stats["rows"] / len(group)})
    tables["monthly"] = pd.DataFrame(monthly_rows)
    tables["split_summary"] = pd.DataFrame(split_rows)
    tables["age_group"] = pd.DataFrame(age_rows)
    tables["monthly_age_group"] = pd.DataFrame(monthly_age_rows)

    feature_rows, psi_rows, missing_rows = [], [], []
    train_mask = clean["month"].le(5)
    for month, group in clean.groupby("month", sort=True):
        for column in MINUS_ONE_MISSING + NEGATIVE_MISSING:
            missing_rows.append({"month": int(month), "feature": column,
                                 "missing_count": int(group[column].isna().sum()),
                                 "missing_fraction": float(group[column].isna().mean())})
        for feature in KEY_NUMERIC:
            series = group[feature]
            quantiles = series.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            feature_rows.append({"month": int(month), "feature": feature, "observed_count": int(series.notna().sum()),
                                 "missing_fraction": float(series.isna().mean()), "mean": float(series.mean()),
                                 "std": float(series.std()), "p10": quantiles.loc[0.1], "p25": quantiles.loc[0.25],
                                 "median": quantiles.loc[0.5], "p75": quantiles.loc[0.75], "p90": quantiles.loc[0.9]})
        for feature in KEY_NUMERIC + CATEGORICAL:
            psi, bins = calculate_psi(clean.loc[train_mask, feature], group[feature], feature in CATEGORICAL)
            psi_rows.append({"month": int(month), "split": split_for_month(int(month)), "feature": feature,
                             "psi": psi, "bins_including_missing": bins, "reference": "pooled months 0-5",
                             "binning": fit_psi_bins(clean.loc[train_mask, feature], feature in CATEGORICAL)["kind"]})
    tables["monthly_feature_summary"] = pd.DataFrame(feature_rows)
    tables["monthly_psi"] = pd.DataFrame(psi_rows)
    tables["monthly_missing"] = pd.DataFrame(missing_rows)

    table_dir, figure_dir = output_dir / "tables", output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(table_dir / f"eda_{name}.csv", index=False, encoding="utf-8-sig")
    save_plots(tables, figure_dir)
    cache_verification = None
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(cache_path, index=False, compression="zstd")
        check_columns = ["fraud_bool", "month", "intended_balcon_amount", "prev_address_months_count", "velocity_6h", "credit_risk_score"]
        roundtrip = pd.read_parquet(cache_path, columns=check_columns)
        if not roundtrip.equals(raw[check_columns]):
            raise ValueError("Parquet roundtrip altered source values.")
        cache_verification = {"rows": len(roundtrip), "columns": len(raw.columns),
                              "roundtrip_columns": check_columns, "exact_match": True}

    constants = tables["schema"].loc[tables["schema"]["constant"], "column"].tolist()
    summary = {
        "source_path": str(input_path.resolve()), "source_bytes": input_path.stat().st_size,
        "rows": len(raw), "columns": len(raw.columns), "raw_memory_bytes": raw_memory_bytes,
        "fraud": rate_stats(raw), "native_missing_cells": native_missing_count,
        "duplicate_rows_including_label_and_month": duplicate_rows,
        "constant_columns": constants,
        "month_values": sorted(int(v) for v in raw["month"].unique()),
        "month_is_monotonic_in_source": bool(raw["month"].is_monotonic_increasing),
        "split_months": {k: list(v) for k, v in SPLITS.items()},
        "missing_rules": {**{c: "value == -1" for c in MINUS_ONE_MISSING},
                          **{c: "value < 0" for c in NEGATIVE_MISSING}},
        "negative_values_retained": ["velocity_6h", "credit_risk_score"],
        "sentinel_missing_cells_excluding_native_nulls": int(tables["schema"]["effective_missing_count"].sum()) - native_missing_count,
        "missing_by_feature": tables["schema"].query("effective_missing_count > 0")[["column", "effective_missing_count", "effective_missing_fraction"]].to_dict("records"),
        "monthly": tables["monthly"].to_dict("records"),
        "split_summary": tables["split_summary"].to_dict("records"),
        "age_groups": tables["age_group"].to_dict("records"),
        "test_month_top_feature_psi": tables["monthly_psi"].query("month == 7").sort_values("psi", ascending=False).to_dict("records"),
        "parquet_cache_path": str(cache_path.resolve()) if cache_path is not None else None,
        "parquet_cache_bytes": cache_path.stat().st_size if cache_path is not None else None,
        "parquet_cache_verified": cache_verification,
        "parquet_cache_values": "Original source values; documented sentinels are preserved.",
        "psi_method": {"version": 2, "numeric_discrete_max_values": PSI_LOW_CARDINALITY_MAX,
                       "bin_fitting": "reference training rows only", "pseudocount": 0.5,
                       "discrete": "one bin per training value plus reserved unseen and missing bins",
                       "continuous": "training deciles with infinite outer edges plus missing bin"},
        "notes": [
            "All 1,000,000 Base rows are included; no sampling is used for tables or figures.",
            "The source CSV is never modified. Documented missing values are masked only in descriptive calculations.",
            "Official datasheet Q7 explicitly defines all negative intended_balcon_amount values as missing.",
            "The datasheet's device_distinct_emails corresponds to CSV field device_distinct_emails_8w.",
            "Valid negative velocity_6h and credit_risk_score values are retained.",
            "month is a split variable and must be excluded from model features; it may still proxy population changes in other features.",
            "Age is rounded to decades; age >= 50 is the observed bin-based group definition, not exact chronological age.",
            "Month 6 is validation and month 7 is the final test month. EDA does not inspect model test predictions.",
            "All-data label summaries are descriptive only and must not guide model or threshold selection.",
            "PSI bins are defined by training data only: categories and numeric fields with <=20 distinct training values use value bins plus unseen/missing bins; continuous fields use deciles plus a missing bin; every bin has a 0.5 pseudocount. PSI is descriptive, not a formal significance test.",
            "No model imputation, scaling, encoding, or feature selection is fitted by this module.",
        ],
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "tables": [str((table_dir / f"eda_{name}.csv").resolve()) for name in tables],
        "figures": [str(p.resolve()) for p in sorted(figure_dir.glob("eda_*.png"))],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eda_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"rows": len(raw), "fraud_rate": summary["fraud"]["fraud_rate"], "constant_columns": constants,
                      "native_missing_cells": native_missing_count, "missing_by_feature": summary["missing_by_feature"],
                      "outputs": str(output_dir.resolve())}, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/Base.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/processed/Base.parquet")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--expected-rows", type=int, default=1_000_000)
    args = parser.parse_args()
    analyze(args.input, args.output_dir, None if args.no_cache else args.cache, args.expected_rows)


if __name__ == "__main__":
    main()
