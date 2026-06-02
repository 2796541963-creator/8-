import torch.nn as nn
from transformers import BertModel
from config import Config

config = Config()


class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.ModuleList(
            [
                nn.Linear(
                    self.bert.config.hidden_size, config.num_classes["fault_type"]
                ),
                nn.Linear(
                    self.bert.config.hidden_size, config.num_classes["risk_level"]
                ),
                nn.Linear(
                    self.bert.config.hidden_size, config.num_classes["department"]
                ),
            ]
        )

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(pooled_output)
        return [fc(pooled_output) for fc in self.fc]
    
if __name__ == "__main__":
    model = BertClassifier()
    print(model)