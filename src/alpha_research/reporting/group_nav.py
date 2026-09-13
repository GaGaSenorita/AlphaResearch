"""Daily factor-sort NAV diagnostics, using the thesis close-to-close label.

These are retrospective, frictionless label portfolios, not execution backtests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .output import FIGURE_FORMATS, save_figure

REPO_ROOT = Path(__file__).resolve().parents[3]


def compute_group_returns(panel: pd.DataFrame, factor: str, groups: int = 10):
    """Sort valid pairs ascending each day; equal weight within each group.

    Ties are resolved by instrument ID, independent of row order and labels.
    Missing/nonfinite pairs are omitted, consistent with the thesis evaluator.
    Dates unable to form all groups are recorded and excluded, never zero-filled.
    """
    if groups < 2:
        raise ValueError("At least two groups are required")
    if panel.index.has_duplicates:
        raise ValueError("Duplicate stock/date observations")
    rows, skipped = [], []
    for date, frame in panel[[factor, "LABEL"]].groupby(level="datetime", sort=True):
        clean = frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index()
        if len(clean) < groups or clean[factor].nunique() < 2:
            skipped.append({"date": str(date.date()), "valid_pairs": len(clean)})
            continue
        rank_ic = clean[factor].rank().corr(clean["LABEL"].rank())
        clean = clean.sort_values([factor, "instrument"], kind="stable")
        clean["group"] = np.arange(len(clean)) * groups // len(clean) + 1
        means = clean.groupby("group")["LABEL"].mean()
        row = {"datetime": date, "valid_pairs": len(clean), "rank_ic": rank_ic,
               "missing_pairs": len(frame) - len(clean),
               "tied_observations": int(clean[factor].duplicated(keep=False).sum())}
        for group in range(1, groups + 1):
            row[f"Group-{group}"] = float(means.loc[group])
            row[f"count_{group}"] = int((clean["group"] == group).sum())
        row["Long-Short"] = row[f"Group-{groups}"] - row["Group-1"]
        rows.append(row)
    if not rows:
        raise ValueError("No dates with sufficient valid observations")
    return pd.DataFrame(rows).set_index("datetime"), skipped


def compound_nav(returns: pd.DataFrame) -> pd.DataFrame:
    """One unit initial capital, daily compounding; LS is +100% top/-100% bottom."""
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Cannot compound nonfinite returns")
    if (returns <= -1).any().any():
        raise ValueError("A portfolio loses at least 100%; compounding is undefined")
    return (1 + returns).cumprod()


def plot_nav(daily, factor, prefix, *, groups, split_lines=None, output_format="pdf",
             write_csv=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    columns = [f"Group-{i}" for i in range(1, groups + 1)] + ["Long-Short"]
    nav = compound_nav(daily[columns])
    nav.index = pd.DatetimeIndex(daily["return_date"])
    nav.index.name = "return_date"
    # Signal at first close; first return accrues at the following trading close.
    nav = pd.concat([pd.DataFrame(1., index=[daily.index[0]], columns=columns), nav])
    nav.index.name = "return_date"
    if write_csv:
        prefix.parent.mkdir(parents=True, exist_ok=True)
        nav.to_csv(prefix.with_suffix(".csv"), float_format="%.12g")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(12, 6.3))
        ax2 = ax.twinx()
        ax2.fill_between(nav.index, 0, nav["Long-Short"], color="#bcbcbc", alpha=.58)
        ax2.plot(nav.index, nav["Long-Short"], color="#aaaaaa", lw=.7)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        colors = plt.get_cmap("tab10").colors
        lines = [ax.plot(nav.index, nav[col], label=col, color=colors[i % 10], lw=1.15)[0]
                 for i, col in enumerate(columns[:-1])]
        for date, label in (split_lines or {}).items():
            date = pd.Timestamp(date)
            if nav.index.min() < date < nav.index.max():
                ax.axvline(date, color="#555555", lw=.8, ls="--", alpha=.6)
                ax.text(date, .98, " " + label, transform=ax.get_xaxis_transform(),
                        va="top", ha="left", fontsize=9, color="#555555")
        ax.legend(handles=lines + [Patch(facecolor="#bcbcbc", alpha=.58,
                  label=f"Long-Short (G{groups} - G1)")], loc="upper left", ncol=2,
                  fontsize=8.5, framealpha=.9)
        ax.set_ylabel("Group NAV (initial = 1)")
        ax2.set_ylabel("Long-short NAV (initial = 1; gross exposure = 2)")
        ax.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.25)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlabel("Return realization date")
        fig.suptitle(factor["title"], fontsize=15, y=.97)
        ax.set_title(f"Group NAV Performance | signal dates {daily.index.min():%Y-%m-%d}"
                     f" to {daily.index.max():%Y-%m-%d} | fee rate = 0", fontsize=10, pad=12)
        fig.text(.5, .017, f"CSI 300 dated membership | G1 = lowest; G{groups} = highest | "
                 "Daily equal-weight sorting | Retrospective close-to-close diagnostic",
                 ha="center", fontsize=8.5, color="#555555")
        fig.subplots_adjust(left=.075, right=.92, bottom=.13, top=.865)
        save_figure(fig, prefix, output_format=output_format, dpi=200,
                    bbox_inches="tight", pad_inches=.12)
        plt.close(fig)
    return {k: float(v) for k, v in nav.iloc[-1].items()}


def render_saved(source: Path, output: Path, *, output_format="pdf") -> None:
    """Replot preserved daily CSVs without loading Qlib or rewriting evidence."""
    metadata_path = source / "metadata.json"
    recovery = "Restore the preserved NAV files or use --recompute --provider PATH."
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing saved NAV metadata: {metadata_path}. {recovery}")
    metadata = json.loads(metadata_path.read_text())
    config = metadata["config"]
    groups = config["groups"]
    columns = [f"Group-{i}" for i in range(1, groups + 1)] + ["Long-Short"]
    datasets = []
    # Verify every input before replacing any figure. Recompounding rounded daily
    # CSVs is allowed only within their export precision of the saved NAVs.
    for factor in config["factors"]:
        name = factor["name"]
        path = source / f"{name.lower()}_daily.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing saved daily CSV: {path}. {recovery}")
        daily = pd.read_csv(path, index_col="datetime", parse_dates=["datetime", "return_date"])
        if (daily.empty or daily.index.has_duplicates or not daily.index.is_monotonic_increasing
                or daily["return_date"].isna().any()):
            raise ValueError(f"Invalid saved daily dates: {path}")
        record = metadata["factors"][name]
        if len(daily) != record["daily_count"]:
            raise ValueError(f"Saved daily count differs from metadata: {path}")
        for split, bounds in config["splits"].items():
            mean_ic = daily.loc[slice(*bounds), "rank_ic"].mean()
            expected = record["split_metrics"][split]["rank_ic"]
            if not np.isfinite(mean_ic) or not np.isclose(mean_ic, expected, rtol=1e-9, atol=1e-10):
                raise ValueError(f"Saved {name} {split} RankIC differs from metadata")
        for period, sample in (("full", daily), ("test", daily.loc[slice(*config["splits"]["test"])])):
            if sample.empty:
                raise ValueError(f"No saved {period} observations for {name}")
            terminal = compound_nav(sample[columns]).iloc[-1]
            expected = pd.Series(record[f"{period}_terminal_nav"])[columns]
            if not np.allclose(terminal, expected, rtol=1e-9, atol=1e-9):
                raise ValueError(f"Saved {name} {period} NAV differs from metadata")
        datasets.append((factor, daily))
    split_lines = {config["splits"]["validation"][0]: "Validation", config["splits"]["test"][0]: "Test"}
    for factor, daily in datasets:
        for period, sample in (("full", daily), ("test", daily.loc[slice(*config["splits"]["test"])])):
            plot_nav(sample, factor, output / f"{factor['name'].lower()}_{period}_nav",
                     groups=groups, split_lines=split_lines if period == "full" else None,
                     output_format=output_format, write_csv=False)
    print(f"Rendered saved NAV figures to {output}; source CSVs and metadata unchanged", flush=True)


def recompute(config: dict, provider: Path, output: Path, *, workers=4, output_format="pdf") -> None:
    """Re-evaluate raw factor expressions using a local Qlib market-data provider."""
    output.mkdir(parents=True, exist_ok=True)
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D
    qlib.init(provider_uri=str(provider), region=REG_CN, kernels=max(1, workers),
              joblib_backend="threading")
    factors = config["factors"]
    fields = [f["expression"] for f in factors] + [config["label_expression"]]
    print("Loading factor panels from the local Qlib provider...", flush=True)
    panel = D.features(D.instruments(config["market"]), fields,
                       start_time=config["start"], end_time=config["end"], freq="day")
    panel.columns = [f["name"] for f in factors] + ["LABEL"]
    calendar = pd.DatetimeIndex(D.calendar(freq="day"))
    next_day = dict(zip(calendar[:-1], calendar[1:]))
    metadata = {"config": config, "provider": str(provider), "qlib_version": qlib.__version__,
                "fee_rate": 0, "long_short": f"daily return = G{config['groups']} - G1; +100% long, -100% short; daily compounding",
                "timing": "Signal at t close; label C[t+1]/C[t]-1; NAV plotted at t+1. Not an executable backtest.",
                "sorting": "Ascending valid-pair factor values; ties broken by instrument ID; equal weights; no factor sign reversal.",
                "full_period": "Continuous calendar, including days between formal evaluation splits.",
                "provider_file_sha256": {}, "factors": {}}
    for relative in ("calendars/day.txt", f"instruments/{config['market']}.txt"):
        metadata["provider_file_sha256"][relative] = hashlib.sha256((provider / relative).read_bytes()).hexdigest()
    summary_rows = []
    for factor in factors:
        name = factor["name"]
        print(f"Computing daily deciles: {name}", flush=True)
        daily, skipped = compute_group_returns(panel, name, config["groups"])
        daily["return_date"] = daily.index.map(next_day)
        if daily["return_date"].isna().any():
            raise ValueError("Provider calendar lacks a return realization date")
        daily.to_csv(output / f"{name.lower()}_daily.csv", float_format="%.12g")
        record = {"skipped_dates": skipped, "daily_count": len(daily), "split_metrics": {}}
        for split, (start, end) in config["splits"].items():
            sample = daily.loc[start:end]
            mean_ic = float(sample["rank_ic"].mean())
            expected = factor["expected_rank_ic"][split]
            if abs(mean_ic - expected) > 0.000051:
                raise ValueError(f"{name} {split} RankIC mismatch: {mean_ic} vs {expected}")
            metrics = {"rank_ic": mean_ic, "valid_ic_days": int(sample["rank_ic"].count()),
                       "daily_spread_mean": float(sample["Long-Short"].mean()),
                       "min_valid_pairs": int(sample["valid_pairs"].min())}
            record["split_metrics"][split] = metrics
            summary_rows.append({"factor": name, "split": split, **metrics})
        for period, sample in (("full", daily), ("test", daily.loc[slice(*config["splits"]["test"])])):
            prefix = output / f"{name.lower()}_{period}_nav"
            record[f"{period}_terminal_nav"] = plot_nav(sample, factor, prefix,
                groups=config["groups"], split_lines={"2021-01-01": "Validation", "2022-01-01": "Test"}
                if period == "full" else None, output_format=output_format)
        metadata["factors"][name] = record
    pd.DataFrame(summary_rows).to_csv(output / "summary.csv", index=False)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Saved figures, NAV series, daily returns and verified RankIC to {output}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true",
                        help="recompute factors from raw Qlib data instead of plotting saved daily CSVs")
    parser.add_argument("--source-dir", type=Path,
                        help="saved daily CSVs and metadata (default: figures/representative_factor_nav)")
    parser.add_argument("--config", type=Path, help="factor config for --recompute")
    parser.add_argument("--provider", type=Path, help="Qlib provider for --recompute")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "figures/representative_factor_nav")
    parser.add_argument("--workers", type=int, default=4, help="Qlib workers for --recompute")
    parser.add_argument("--format", choices=FIGURE_FORMATS, default="pdf",
                        help="one output image format (default: pdf)")
    args = parser.parse_args(argv)
    if not args.recompute and (args.config is not None or args.provider is not None):
        parser.error("--config and --provider require --recompute; saved plots use their metadata config")
    if args.recompute and args.source_dir is not None:
        parser.error("--source-dir is only used when plotting saved daily CSVs")
    output = args.output_dir.expanduser().resolve()
    if args.recompute:
        config_path = args.config or REPO_ROOT / "configs/representative_factor_nav.json"
        config = json.loads(config_path.expanduser().read_text())
        provider = (args.provider or REPO_ROOT.parent / "quantaalpha_qlib_csi300/cn_data").expanduser().resolve()
        recompute(config, provider, output, workers=args.workers, output_format=args.format)
    else:
        source = (args.source_dir or REPO_ROOT / "figures/representative_factor_nav").expanduser().resolve()
        render_saved(source, output, output_format=args.format)
    return 0
