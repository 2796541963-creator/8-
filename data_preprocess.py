import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

from config import Config

config = Config()

label2id_dict = eval(open(config.label2id_path, "r", encoding="utf-8").read())
label2id_list = [value for key, value in label2id_dict.items()]

tokenizer = BertTokenizer.from_pretrained(config.bert_path)


def org_data_preprocess():
    data = pd.read_csv(
        config.orig_data_path,
        sep="\t",
        header=None,
        names=["text", "fault_type", "risk_level", "department"],
    )
    data["fault_type"] = data["fault_type"].map(label2id_dict["fault_type"])
    data["risk_level"] = data["risk_level"].map(label2id_dict["risk_level"])
    data["department"] = data["department"].map(label2id_dict["department"])

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=24)
    _, dev_data = train_test_split(train_data, test_size=0.1, random_state=24)

    train_data.to_csv(config.train_path, sep="\t", index=False, header=False)
    test_data.to_csv(config.test_path, sep="\t", index=False, header=False)
    dev_data.to_csv(config.dev_path, sep="\t", index=False, header=False)


# 读取原始数据
def read_raw_data(data_path):
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            text = parts[0]
            labels = [int(label) for label in parts[1:]]
            data.append((text, labels))
    return data


# 构建数据集
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, labels = self.data[idx]
        return text, labels


def collate_fn(batch):
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    labels = torch.tensor(labels, dtype=torch.long)

    return input_ids, attention_mask, labels


def build_dataloader():
    train_data = read_raw_data(config.train_path)
    test_data = read_raw_data(config.test_path)
    dev_data = read_raw_data(config.dev_path)

    train_dataset = TextDataset(train_data)
    test_dataset = TextDataset(test_data)
    dev_dataset = TextDataset(dev_data)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_dataloader, test_dataloader, dev_dataloader


if __name__ == "__main__":
    train_loader, test_loader, dev_loader = build_dataloader()
    for input_ids, attention_mask, labels in train_loader:
        print("input_ids shape:", input_ids.shape)
        print("attention_mask shape:", attention_mask.shape)
        print("labels shape:", labels.shape)
        break
