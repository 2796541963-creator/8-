import torch.nn as nn
from config import Config
from transformers import BertModel
config = Config()


class Moudle(nn.Module):
    def __init__(self,fault_num,risk_grad_num,department_num):
        super(Moudle,self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.fault =nn.Linear(config.hidden_size,fault_num)
        self.risk_grad=nn.Linear(config.hidden_size,risk_grad_num)
        self.department= nn.Linear(config.hidden_size,department_num)
    def forward(self,input_ids,attention_mask):
        _,pooled = self.bert(input_ids = input_ids,attention_mask = attention_mask,return_dict=False)
        fault_out = self.fault(pooled)
        risk_grad_out = self.risk_grad(pooled)
        department_out = self.department(pooled)
        return fault_out,risk_grad_out,department_out





