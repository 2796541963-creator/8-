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
    torch.load(config.model_save_path + "/bert_classifier_2026-06-04_22-18-45.pth")
)
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
        outputs = model(input_ids, attention_mask)
        fault_type_preds.extend([torch.argmax(output[0], dim=-1) for output in outputs])
        risk_level_preds.extend([torch.argmax(output[1], dim=-1) for output in outputs])
        department_preds.extend([torch.argmax(output[2], dim=-1) for output in outputs])
        fault_type_true.extend(labels[:, 0])
        risk_level_true.extend(labels[:, 1])
        department_true.extend(labels[:, 2])

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
