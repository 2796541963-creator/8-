from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "configs" / "labels.json"
SPLIT_FILES = ["train.tsv", "val.tsv", "test.tsv"]
LABEL_COLUMNS = ["fault_category", "risk_level", "department"]
RISK_ORDER = ["P0", "P1", "P2", "P3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build label mappings for classification tasks.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory with train/val/test TSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to labels.json.")
    return parser.parse_args()


def read_split_files(input_dir: Path) -> dict[str, pd.DataFrame]:
    # 标签映射必须来自已经切好的数据集，保证训练、验证、测试使用同一套标签空间
    splits: dict[str, pd.DataFrame] = {}
    for file_name in SPLIT_FILES:
        path = input_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")

        split_name = path.stem
        df = pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8-sig", keep_default_na=False)
        missing_columns = [column for column in LABEL_COLUMNS if column not in df.columns]
        if missing_columns:
            raise ValueError(f"{path} missing label columns: {missing_columns}")
        splits[split_name] = df
    return splits


def ordered_labels(values: pd.Series, column: str) -> list[str]:
    # 风险等级有明确业务顺序，P0 最高风险，应固定为 P0/P1/P2/P3
    unique_labels = set(values.tolist())
    if column == "risk_level":
        unknown = unique_labels - set(RISK_ORDER)
        if unknown:
            raise ValueError(f"Unknown risk labels: {sorted(unknown)}")
        return [label for label in RISK_ORDER if label in unique_labels]

    # 其他标签没有天然大小关系，使用字典序保证每次生成结果稳定
    return sorted(unique_labels)


def build_mapping(labels: list[str]) -> dict[str, dict[str, int] | dict[str, str] | int]:
    # label2id 用于训练时把字符串标签转成整数；id2label 用于推理时把整数结果转回中文标签
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {str(index): label for label, index in label2id.items()}
    return {
        "num_labels": len(labels),
        "label2id": label2id,
        "id2label": id2label,
    }


def build_labels_config(splits: dict[str, pd.DataFrame]) -> dict[str, object]:
    merged = pd.concat(splits.values(), ignore_index=True)
    config: dict[str, object] = {
        "version": 1,
        "splits": {name: len(df) for name, df in splits.items()},
        "tasks": {},
    }

    tasks: dict[str, object] = {}
    for column in LABEL_COLUMNS:
        labels = ordered_labels(merged[column], column)
        tasks[column] = build_mapping(labels)
    config["tasks"] = tasks
    return config


def print_summary(config: dict[str, object]) -> None:
    print("\n=== Label config summary ===")
    print(f"version: {config['version']}")
    print(f"splits: {config['splits']}")

    tasks = config["tasks"]
    if not isinstance(tasks, dict):
        return

    for task_name, task_config in tasks.items():
        if not isinstance(task_config, dict):
            continue
        print(f"\n[{task_name}]")
        print(f"num_labels: {task_config['num_labels']}")
        label2id = task_config["label2id"]
        if isinstance(label2id, dict):
            for label, label_id in label2id.items():
                print(f"{label}: {label_id}")


def main() -> None:
    args = parse_args()
    splits = read_split_files(args.input_dir)
    config = build_labels_config(splits)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print_summary(config)
    print(f"\nLabels saved to: {args.output}")


if __name__ == "__main__":
    main()
