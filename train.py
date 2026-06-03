import torch
import torch.nn as nn
from config import Config
from data_preprocess import build_dataloader
from bert_model import BertClassifier
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")

config = Config()

train_loader, test_loader, dev_loader = build_dataloader()
model = BertClassifier().to(config.device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


def model2train():
    model.train()
    for epoch in range(config.num_epochs):
        total_loss = 0
        for input_ids, attention_mask, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}"
        ):
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            labels = labels.to(config.device)

            outputs = model(input_ids, attention_mask)
            loss = sum(
                criterion(output, labels[:, i]) for i, output in enumerate(outputs)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{config.num_epochs}, Loss: {avg_loss:.4f}")


def model2dev():
    model.eval()
    task_names = ["fault_type", "risk_level", "department"]
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(dev_loader, desc="Dev"):
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            labels = labels.to(config.device)

            y_preds = model(input_ids, attention_mask)
            y_pred_list = [torch.argmax(y_pred, dim=-1) for y_pred in y_preds]
            y_pred_list = torch.stack(y_pred_list, dim=1)

            all_preds.append(y_pred_list.cpu())
            all_trues.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_trues = torch.cat(all_trues, dim=0)

    for i, name in enumerate(task_names):
        y_true = all_trues[:, i].numpy()
        y_pred = all_preds[:, i].numpy()
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        print(f"[{name}] Accuracy: {acc:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


if __name__ == "__main__":
    model2train()
    model2dev()
