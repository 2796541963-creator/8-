from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COLUMNS = ["text", "fault", "risk_grad", "department"]
LABEL_COLUMNS = ["fault", "risk_grad", "department"]
DEFAULT_FILE_PATH = Path(__file__).resolve().parent / "data" / "train.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the repair text training dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_FILE_PATH,
        help="Path to the TSV data file. Defaults to data/train.txt.",
    )
    return parser.parse_args()


def read_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    if file_path.is_dir():
        raise IsADirectoryError(f"Expected a data file, but got a directory: {file_path}")

    return pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=COLUMNS,
        dtype={"text": str, "fault": str, "risk_grad": str, "department": str},
        encoding="utf-8-sig",
        keep_default_na=False,
    )


def print_label_distribution(data: pd.DataFrame, column: str) -> None:
    counts = data[column].value_counts(dropna=False)
    print(f"\n{column} 标签分布：")
    for label, count in counts.items():
        percent = count / len(data) * 100 if len(data) else 0
        print(f"标签 {label}：{count} 次，占比 {percent:.2f}%")


def main() -> None:
    args = parse_args()
    data = read_data(args.input)

    print(f"数据文件：{args.input}")
    print("前5行数据：")
    print(data.head(5))
    print(f"总数据量：{len(data)} 行")
    print(f"空字段数量：{int((data == '').sum().sum())}")
    print(f"重复文本数量：{int(data.duplicated(subset=['text']).sum())}")

    for column in LABEL_COLUMNS:
        print_label_distribution(data, column)

    data["text_length"] = data["text"].str.len()
    print("\n文本长度前10行：")
    print(data[["text", "text_length"]].head(10))
    print("\n文本长度统计：")
    print(f"平均长度：{data['text_length'].mean():.2f} 字符")
    print(f"长度标准差：{data['text_length'].std():.2f} 字符")
    print(f"最大长度：{data['text_length'].max()} 字符")
    print(f"最小长度：{data['text_length'].min()} 字符")


if __name__ == "__main__":
    main()
