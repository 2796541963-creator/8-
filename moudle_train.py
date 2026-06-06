import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score

from config import Config
from moudle import Moudle
from utils import dataloader, load_raw_data


TASKS = ("fault", "risk_grad", "department")


def _unpack_batch(batch, device):
    input_ids, attention_mask, fault_labels, risk_grad_labels, department_labels = [
        item.to(device) for item in batch
    ]
    labels = (fault_labels, risk_grad_labels, department_labels)
    return input_ids, attention_mask, labels


def _compute_loss(outputs, labels, loss_fn):
    losses = [loss_fn(output, label) for output, label in zip(outputs, labels)]
    return sum(losses)


def _append_predictions(outputs, labels, preds, reals):
    for task_name, output, label in zip(TASKS, outputs, labels):
        task_preds = torch.argmax(output, dim=1)
        preds[task_name].extend(task_preds.detach().cpu().numpy().tolist())
        reals[task_name].extend(label.detach().cpu().numpy().tolist())


def _metric_summary(preds, reals):
    metrics = {}
    f1_values = []
    for task_name in TASKS:
        y_true = reals[task_name]
        y_pred = preds[task_name]
        metrics[task_name] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        }
        f1_values.append(metrics[task_name]["f1"])
    metrics["avg_f1"] = sum(f1_values) / len(f1_values)
    return metrics


def evaluate(model, data_loader, loss_fn, device, desc="Evaluating"):
    model.eval()
    total_loss = 0.0
    preds = {task_name: [] for task_name in TASKS}
    reals = {task_name: [] for task_name in TASKS}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc):
            input_ids, attention_mask, labels = _unpack_batch(batch, device)
            outputs = model(input_ids, attention_mask)
            loss = _compute_loss(outputs, labels, loss_fn)
            total_loss += loss.item()
            _append_predictions(outputs, labels, preds, reals)

    metrics = _metric_summary(preds, reals)
    metrics["loss"] = total_loss / max(len(data_loader), 1)
    return metrics


def print_metrics(prefix, metrics):
    print(f"{prefix} Loss: {metrics['loss']:.4f}")
    print(f"{prefix} Avg Macro F1: {metrics['avg_f1']:.4f}")
    for task_name in TASKS:
        task_metrics = metrics[task_name]
        print(
            f"{prefix} {task_name}: "
            f"acc={task_metrics['accuracy']:.4f}, "
            f"f1={task_metrics['f1']:.4f}, "
            f"precision={task_metrics['precision']:.4f}"
        )


def moudle_train():

    config = Config()
    _, fault_num, risk_grad_num, department_num = load_raw_data()
    train_loader, val_loader, test_loader = dataloader()
    best_dev_f1 = 0.0
    device = config.device
    epochs = config.num_epochs
    model = Moudle(fault_num, risk_grad_num, department_num).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        train_preds = {task_name: [] for task_name in TASKS}
        train_reals = {task_name: [] for task_name in TASKS}
        for batch in tqdm(train_loader, desc=f"Bert Classifier Training Epoch {epoch + 1}/{epochs}...."):
            input_ids, attention_mask, labels = _unpack_batch(batch, device)
            outputs = model(input_ids, attention_mask)
            loss = _compute_loss(outputs, labels, loss_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _append_predictions(outputs, labels, train_preds, train_reals)

        train_metrics = _metric_summary(train_preds, train_reals)
        train_metrics["loss"] = total_loss / max(len(train_loader), 1)
        val_metrics = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            desc=f"Bert Classifier Validation Epoch {epoch + 1}/{epochs}....",
        )

        print(f"Epoch {epoch + 1}/{epochs}")
        print_metrics("Train", train_metrics)
        print_metrics("Val", val_metrics)

        if val_metrics["avg_f1"] > best_dev_f1:
            best_dev_f1 = val_metrics["avg_f1"]
            save_dir = os.path.dirname(config.model_save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), config.model_save_path)
            print(f"Best model saved to {config.model_save_path}")

    if os.path.exists(config.model_save_path):
        model.load_state_dict(torch.load(config.model_save_path, map_location=device))
    test_metrics = evaluate(model, test_loader, loss_fn, device, desc="Bert Classifier Testing....")
    print_metrics("Test", test_metrics)

if __name__ == '__main__':
    moudle_train()






