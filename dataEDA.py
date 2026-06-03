import pandas as pd
from config import Config

config = Config()


def dataEDA():

    data = pd.read_csv(
        config.orig_data_path,
        sep="\t",
        header=None,
        names=["text", "fault_type", "risk_level", "department"],
    )
    print("数据集基本信息：")
    print(data.info())
    print("\n数据集前5行：")
    print(data.head())

    print("故障类型分布：")
    print(data["fault_type"].value_counts())
    print("风险等级分布：")
    print(data["risk_level"].value_counts())
    print("部门分布：")
    print(data["department"].value_counts())

    print("文本长度分布：")
    data["text_length"] = data["text"].apply(len)
    print(data["text_length"].mean())
    print(data["text_length"].std())
    print(data["text_length"].max())
    print(data["text_length"].min())

    print(
        f"最优文本长度为: {data['text_length'].mean() + 3 * data['text_length'].std():.0f}"
    )


if __name__ == "__main__":
    dataEDA()
