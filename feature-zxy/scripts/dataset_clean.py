from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "manufacturing_repair_text_dataset_cn.txt"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_dataset.tsv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "reports" / "clean_summary.md"
COLUMNS = ["text", "fault_category", "risk_level", "department"]


LEAKAGE_PATTERNS = [
    ("risk_level_hint", re.compile(r"风险等级建议为\s*P[0-3]")),
    ("risk_level_hint", re.compile(r"风险等级[:：]?\s*P[0-3]")),
    ("risk_level_hint", re.compile(r"建议按\s*P[0-3]\s*工单处理")),
    ("risk_level_hint", re.compile(r"按\s*P[0-3]\s*工单处理")),
    ("department_hint", re.compile(r"请尽快安排(?:设备保养/备件管理|工艺/设备工程师|质量/工艺/设备|公辅/设备维修|自动化工程师|自动化/仪表|设备保养|电气维修|机械维修|安全/EHS)")),
    ("department_hint", re.compile(r"请安排(?:设备保养/备件管理|工艺/设备工程师|质量/工艺/设备|公辅/设备维修|自动化工程师|自动化/仪表|设备保养|电气维修|机械维修|安全/EHS)")),
    ("department_hint", re.compile(r"请(?:设备保养/备件管理|工艺/设备工程师|质量/工艺/设备|公辅/设备维修|自动化工程师|自动化/仪表|设备保养|电气维修|机械维修|安全/EHS)确认")),
    ("fault_category_hint", re.compile(r"属于预防性维护任务")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean leakage phrases from repair text dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Path to raw TSV dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path for cleaned TSV dataset.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH, help="Path for cleaning report.")
    parser.add_argument("--preview-rows", type=int, default=10, help="Number of cleaned rows to preview.")
    return parser.parse_args()


def read_dataset(path: Path) -> pd.DataFrame:
    # 原始数据是无表头 TSV，按固定四列读入，避免 pandas 自动猜测列名
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


def normalize_text(text: str) -> str:
    # 统一空白和标点间隙，让后面的正则匹配更稳定
    text = re.sub(r"\s+", "", text.strip())
    text = re.sub(r"[，,；;：:。！？!?]+$", "", text)
    return text


def clean_text(text: str, counter: Counter[str]) -> str:
    # 只删除明显泄漏标签答案的短语，不删除设备、现象、部件等真实故障信息
    cleaned = normalize_text(text)
    for name, pattern in LEAKAGE_PATTERNS:
        cleaned, count = pattern.subn("", cleaned)
        if count:
            counter[name] += count

    cleaned = re.sub(r"[，,；;：:。！？!?]{2,}", "，", cleaned)
    cleaned = re.sub(r"[，,；;：:。！？!?]+$", "", cleaned)
    return cleaned


def remove_invalid_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    # 清洗后如果出现空文本、空标签或重复文本，应从训练数据中移除
    before = len(df)
    df = df.copy()
    df = df[(df[COLUMNS] != "").all(axis=1)]
    after_empty_fields = len(df)
    df = df[df["text"] != ""]
    after_empty_text = len(df)
    df = df.drop_duplicates(subset=COLUMNS, keep="first")
    after_exact_duplicates = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    after_text_duplicates = len(df)

    stats = {
        "removed_empty_fields": before - after_empty_fields,
        "removed_empty_text": after_empty_fields - after_empty_text,
        "removed_exact_duplicates": after_empty_text - after_exact_duplicates,
        "removed_duplicate_text": after_exact_duplicates - after_text_duplicates,
    }
    return df, stats


def build_preview(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, rows: int) -> pd.DataFrame:
    # 预览保留原文本和清洗后文本，方便肉眼判断是否误删关键信息
    preview = raw_df.loc[cleaned_df.index, ["text"]].head(rows).rename(columns={"text": "raw_text"})
    preview["cleaned_text"] = cleaned_df["text"].head(rows).values
    preview["fault_category"] = cleaned_df["fault_category"].head(rows).values
    preview["risk_level"] = cleaned_df["risk_level"].head(rows).values
    preview["department"] = cleaned_df["department"].head(rows).values
    return preview


def write_report(
    path: Path,
    raw_rows: int,
    cleaned_rows: int,
    leakage_counter: Counter[str],
    remove_stats: dict[str, int],
    preview: pd.DataFrame,
) -> None:
    # 报告只写 Markdown，作为清洗过程的审计记录
    leakage_table = pd.DataFrame(
        [{"pattern_group": name, "removed_count": count} for name, count in leakage_counter.items()]
    )
    if leakage_table.empty:
        leakage_table = pd.DataFrame(columns=["pattern_group", "removed_count"])

    remove_table = pd.DataFrame(
        [{"item": name, "count": count} for name, count in remove_stats.items()]
    )

    lines = [
        "# Dataset Clean Summary",
        "",
        f"- raw_rows: {raw_rows}",
        f"- cleaned_rows: {cleaned_rows}",
        f"- removed_rows: {raw_rows - cleaned_rows}",
        "",
        "## Removed Leakage Phrases",
        "",
        leakage_table.to_markdown(index=False),
        "",
        "## Removed Rows",
        "",
        remove_table.to_markdown(index=False),
        "",
        "## Cleaned Preview",
        "",
        preview.to_markdown(index=False),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    raw_df = read_dataset(args.input)
    cleaned_df = raw_df.copy()
    leakage_counter: Counter[str] = Counter()
    cleaned_df["text"] = cleaned_df["text"].map(lambda text: clean_text(text, leakage_counter))
    cleaned_df, remove_stats = remove_invalid_rows(cleaned_df)
    preview = build_preview(raw_df, cleaned_df, args.preview_rows)

    cleaned_df.to_csv(args.output, sep="\t", index=False, encoding="utf-8")
    write_report(args.report, len(raw_df), len(cleaned_df), leakage_counter, remove_stats, preview)

    print("\n=== Clean summary ===")
    print(f"raw_rows: {len(raw_df)}")
    print(f"cleaned_rows: {len(cleaned_df)}")
    print(f"removed_rows: {len(raw_df) - len(cleaned_df)}")
    print("\n=== Removed leakage phrases ===")
    for name, count in leakage_counter.items():
        print(f"{name}: {count}")
    print("\n=== Removed rows ===")
    for name, count in remove_stats.items():
        print(f"{name}: {count}")
    print("\n=== Cleaned preview ===")
    print(preview.to_string(index=False))
    print(f"\nCleaned dataset saved to: {args.output}")
    print(f"Clean report saved to: {args.report}")


if __name__ == "__main__":
    main()
