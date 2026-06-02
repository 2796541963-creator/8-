from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "manufacturing_repair_text_dataset_cn.txt"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "data" / "reports"
COLUMNS = ["text", "fault_category", "risk_level", "department"]
LABEL_COLUMNS = ["fault_category", "risk_level", "department"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA for equipment repair text dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH, help="Path to raw TSV dataset.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR, help="Directory for EDA reports.")
    parser.add_argument("--preview-rows", type=int, default=20, help="Number of rows to preview.")
    return parser.parse_args()


def read_dataset(path: Path) -> pd.DataFrame:
    # 原始数据是无表头 TSV，四列依次为：文本、故障类别、风险等级、负责部门
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=COLUMNS,
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False,
    )


def distribution_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    # 统计单个标签列的样本数和占比，用于观察类别分布
    counts = df[column].value_counts(dropna=False).rename_axis(column).reset_index(name="count")
    counts["ratio"] = counts["count"] / len(df)
    return counts


def balance_summary(df: pd.DataFrame, column: str) -> dict[str, float | int | str]:
    # 最大类 / 最小类越接近 1，说明该标签越均衡
    counts = df[column].value_counts(dropna=False)
    max_count = int(counts.max())
    min_count = int(counts.min())
    return {
        "label": column,
        "class_count": int(counts.size),
        "max_count": max_count,
        "min_count": min_count,
        "max_min_ratio": round(max_count / min_count, 4) if min_count else float("inf"),
        "most_common": str(counts.idxmax()),
        "least_common": str(counts.idxmin()),
    }


def text_length_summary(df: pd.DataFrame) -> pd.DataFrame:
    # 文本长度分布可帮助判断后续 MacBERT 的 max_length 设置是否足够
    text_lengths = df["text"].str.len()
    return text_lengths.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_frame("text_length")


def write_report(
    output_path: Path,
    df: pd.DataFrame,
    preview: pd.DataFrame,
    distributions: dict[str, pd.DataFrame],
    balance: pd.DataFrame,
    combo_distribution: pd.DataFrame,
    length_summary: pd.DataFrame,
) -> None:
    # Markdown 报告作为唯一落盘产物，避免 EDA 阶段产生过多中间文件
    lines = [
        "# Equipment Repair Dataset EDA",
        "",
        f"- dataset_rows: {len(df)}",
        f"- dataset_columns: {len(df.columns)}",
        f"- duplicate_text_rows: {int(df.duplicated(subset=['text']).sum())}",
        f"- empty_field_count: {int((df == '').sum().sum())}",
        "",
        "## Preview",
        "",
        preview.to_markdown(index=False),
        "",
        "## Balance Summary",
        "",
        balance.to_markdown(index=False),
        "",
        "## Text Length Summary",
        "",
        length_summary.to_markdown(),
    ]

    for column, table in distributions.items():
        lines.extend(["", f"## {column} Distribution", "", table.to_markdown(index=False)])

    lines.extend(["", "## Top Label Combinations", "", combo_distribution.head(30).to_markdown(index=False)])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = read_dataset(args.input)
    preview = df.head(args.preview_rows)
    distributions = {column: distribution_table(df, column) for column in LABEL_COLUMNS}
    balance = pd.DataFrame([balance_summary(df, column) for column in LABEL_COLUMNS])
    combo_distribution = (
        df.groupby(LABEL_COLUMNS)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    length_summary = text_length_summary(df)

    write_report(
        args.output_dir / "eda_summary.md",
        df,
        preview,
        distributions,
        balance,
        combo_distribution,
        length_summary,
    )

    print("\n=== Preview: first rows ===")
    print(preview.to_string(index=False))
    print("\n=== Dataset size ===")
    print(f"rows: {len(df)}")
    print(f"columns: {len(df.columns)}")
    print("\n=== Balance summary ===")
    print(balance.to_string(index=False))
    print("\n=== Distributions ===")
    for column, table in distributions.items():
        print(f"\n[{column}]")
        print(table.to_string(index=False))
    print("\n=== Text length summary ===")
    print(length_summary.to_string())
    print(f"\nReport saved to: {args.output_dir / 'eda_summary.md'}")


if __name__ == "__main__":
    main()
