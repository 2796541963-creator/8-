from data_preprocess import build_dataloader
from bert_model import BertClassifier
from config import Config
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import torch

config = Config()
_, test_dataloader, _ = build_dataloader()

model = BertClassifier()
model.load_state_dict(
    torch.load(
        config.model_save_path + "/bert_classifier_2026-06-04_22-18-45.pth",
        map_location=config.device,
    )
)
model.to(config.device)
model.eval()
with torch.no_grad():
    fault_type_preds = []
    risk_level_preds = []
    department_preds = []
    fault_type_true = []
    risk_level_true = []
    department_true = []
    for batch in test_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(config.device)
        attention_mask = attention_mask.to(config.device)
        fault_type_logits, risk_level_logits, department_logits = model(
            input_ids, attention_mask
        )
        fault_type_preds.extend(torch.argmax(fault_type_logits, dim=-1).cpu().tolist())
        risk_level_preds.extend(torch.argmax(risk_level_logits, dim=-1).cpu().tolist())
        department_preds.extend(torch.argmax(department_logits, dim=-1).cpu().tolist())
        fault_type_true.extend(labels[:, 0].tolist())
        risk_level_true.extend(labels[:, 1].tolist())
        department_true.extend(labels[:, 2].tolist())

    print("Fault Type Classification Report:")
    print(classification_report(fault_type_true, fault_type_preds))
    print("Accuracy:", accuracy_score(fault_type_true, fault_type_preds))
    print(
        "Precision:",
        precision_score(fault_type_true, fault_type_preds, average="micro"),
    )
    print("Recall:", recall_score(fault_type_true, fault_type_preds, average="micro"))
    print("F1 Score:", f1_score(fault_type_true, fault_type_preds, average="micro"))
    print("\nRisk Level Classification Report:")
    print(classification_report(risk_level_true, risk_level_preds))
    print("Accuracy:", accuracy_score(risk_level_true, risk_level_preds))
    print(
        "Precision:",
        precision_score(risk_level_true, risk_level_preds, average="micro"),
    )
    print("Recall:", recall_score(risk_level_true, risk_level_preds, average="micro"))
    print("F1 Score:", f1_score(risk_level_true, risk_level_preds, average="micro"))
    print("\nDepartment Classification Report:")
    print(classification_report(department_true, department_preds))
    print("Accuracy:", accuracy_score(department_true, department_preds))
    print(
        "Precision:",
        precision_score(department_true, department_preds, average="micro"),
    )
    print("Recall:", recall_score(department_true, department_preds, average="micro"))
    print("F1 Score:", f1_score(department_true, department_preds, average="micro"))
