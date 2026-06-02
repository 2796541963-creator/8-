from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_dataset.tsv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "reports" / "split_summary.md"
COLUMNS = ["text", "fault_category", "risk_level", "department"]
LABEL_COLUMNS = ["fault_category", "risk_level", "department"]
SPLIT_COLUMN = "split"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split cleaned equipment repair dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Path to cleaned TSV dataset.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for split TSV files.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH, help="Path for split report.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training set ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation set ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split.")
    parser.add_argument("--preview-rows", type=int, default=8, help="Number of rows to preview for each split.")
    return parser.parse_args()


def read_dataset(path: Path) -> pd.DataFrame:
    # cleaned_dataset.tsv 是 dataset_clean.py 的输出，已经带有表头
    if not path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {path}")

    df = pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8-sig", keep_default_na=False)
    missing_columns = [column for column in COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # 只保留训练需要的四列，避免后续误把临时列写入数据集
    return df[COLUMNS].copy()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    # 三个比例应合计为 1；这里允许极小浮点误差
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("Split ratios must be positive.")


def build_stratify_key(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    # 多任务分类要尽量同时保持故障类别、风险等级、部门的组合分布
    return df[columns].astype(str).agg("||".join, axis=1)


def choose_stratify_columns(df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float) -> list[str] | None:
    # 分层键越细，分布保持越好；但如果某个组合样本太少，二次切分会失败
    # 8:1:1 场景下，最小类至少约 10 条，才更稳妥地进入 train/val/test
    min_required = max(2, int(round(1 / min(train_ratio, val_ratio, test_ratio))))
    candidates = [
        ["fault_category", "risk_level", "department"],
        ["fault_category", "risk_level"],
        ["risk_level", "department"],
        ["fault_category"],
        ["risk_level"],
    ]

    for columns in candidates:
        counts = build_stratify_key(df, columns).value_counts()
        if int(counts.min()) >= min_required:
            return columns
    return None


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    stratify_columns: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 先切出训练集，再把剩余部分平均切成验证集和测试集
    # 这样能自然得到 8:1:1，并且每一步都可以使用同一个分层键
    stratify_key = build_stratify_key(df, stratify_columns) if stratify_columns else None
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify_key,
    )

    temp_val_ratio = val_ratio / (val_ratio + test_ratio)
    temp_stratify_key = build_stratify_key(temp_df, stratify_columns) if stratify_columns else None
    val_df, test_df = train_test_split(
        temp_df,
        train_size=temp_val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=temp_stratify_key,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def add_split_name(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    # 报告阶段临时加 split 列，落盘 TSV 不包含该列
    result = df.copy()
    result[SPLIT_COLUMN] = split_name
    return result


def split_size_table(splits: dict[str, pd.DataFrame], total_rows: int) -> pd.DataFrame:
    rows = []
    for split_name, split_df in splits.items():
        rows.append(
            {
                "split": split_name,
                "rows": len(split_df),
                "ratio": round(len(split_df) / total_rows, 6),
            }
        )
    return pd.DataFrame(rows)


def label_distribution_table(splits: dict[str, pd.DataFrame], column: str) -> pd.DataFrame:
    # 每个 split 内部的占比可用于检查 train/val/test 是否保持同样标签结构
    rows = []
    for split_name, split_df in splits.items():
        counts = split_df[column].value_counts().sort_index()
        for label, count in counts.items():
            rows.append(
                {
                    "split": split_name,
                    column: label,
                    "count": int(count),
                    "ratio_in_split": round(int(count) / len(split_df), 6),
                }
            )
    return pd.DataFrame(rows)


def label_combo_table(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # 组合标签用于确认多任务标签的联合分布，尤其适合发现某个 split 是否缺少关键组合
    merged = pd.concat([add_split_name(split_df, name) for name, split_df in splits.items()], ignore_index=True)
    return (
        merged.groupby([SPLIT_COLUMN, *LABEL_COLUMNS])
        .size()
        .reset_index(name="count")
        .sort_values([SPLIT_COLUMN, "count"], ascending=[True, False])
    )


def write_split_files(splits: dict[str, pd.DataFrame], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "train": output_dir / "train.tsv",
        "val": output_dir / "val.tsv",
        "test": output_dir / "test.tsv",
    }
    for split_name, split_df in splits.items():
        split_df.to_csv(output_paths[split_name], sep="\t", index=False, encoding="utf-8")
    return output_paths


def write_report(
    path: Path,
    split_sizes: pd.DataFrame,
    distributions: dict[str, pd.DataFrame],
    combo_distribution: pd.DataFrame,
    previews: dict[str, pd.DataFrame],
    stratify_columns: list[str] | None,
    output_paths: dict[str, Path],
) -> None:
    # Markdown 报告记录本次划分的关键参数、输出路径和分布结果
    lines = [
        "# Dataset Split Summary",
        "",
        f"- stratify_columns: {stratify_columns if stratify_columns else 'None'}",
        f"- train_path: {output_paths['train']}",
        f"- val_path: {output_paths['val']}",
        f"- test_path: {output_paths['test']}",
        "",
        "## Split Sizes",
        "",
        split_sizes.to_markdown(index=False),
    ]

    for column, table in distributions.items():
        lines.extend(["", f"## {column} Distribution", "", table.to_markdown(index=False)])

    lines.extend(["", "## Top Label Combinations", "", combo_distribution.head(60).to_markdown(index=False)])

    for split_name, preview in previews.items():
        lines.extend(["", f"## {split_name} Preview", "", preview.to_markdown(index=False)])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    df = read_dataset(args.input)
    stratify_columns = choose_stratify_columns(df, args.train_ratio, args.val_ratio, args.test_ratio)
    train_df, val_df, test_df = split_dataset(
        df,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
        stratify_columns,
    )

    splits = {"train": train_df, "val": val_df, "test": test_df}
    output_paths = write_split_files(splits, args.output_dir)
    split_sizes = split_size_table(splits, len(df))
    distributions = {column: label_distribution_table(splits, column) for column in LABEL_COLUMNS}
    combo_distribution = label_combo_table(splits)
    previews = {name: split_df.head(args.preview_rows) for name, split_df in splits.items()}
    write_report(args.report, split_sizes, distributions, combo_distribution, previews, stratify_columns, output_paths)

    print("\n=== Split strategy ===")
    print(f"stratify_columns: {stratify_columns if stratify_columns else 'None'}")
    print(f"seed: {args.seed}")
    print("\n=== Split sizes ===")
    print(split_sizes.to_string(index=False))

    for column, table in distributions.items():
        print(f"\n=== {column} distribution ===")
        print(table.to_string(index=False))

    for split_name, preview in previews.items():
        print(f"\n=== {split_name} preview ===")
        print(preview.to_string(index=False))

    print(f"\nTrain saved to: {output_paths['train']}")
    print(f"Validation saved to: {output_paths['val']}")
    print(f"Test saved to: {output_paths['test']}")
    print(f"Split report saved to: {args.report}")


if __name__ == "__main__":
    main()
