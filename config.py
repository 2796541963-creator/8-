import torch
import os
from transformers.models import BertModel,BertTokenizer,BertConfig


class Config(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_path = "./data"
        self.train_path = self.data_path + "/train.txt"
        self.test_path = self.data_path + "/test.txt"
        self.val_path = self.data_path + "/val.txt"
        self.fault_class_path = self.data_path + "/fault_class.txt"
        self.risk_grad_class_path= self.data_path + "/risk_grad_class.txt"
        self.department_class_path= self.data_path + "/department_class_.txt"



        self.orig_data_path = self.data_path +"/manufacturing_repair_text_dataset_cn(2).txt"
        self.num_epochs = 1  # epoch数
        self.batch_size = 256  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = "./bert-base-chinese"  # 预训练BERT模型的路径
        self.bert_model = BertModel.from_pretrained(self.bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)  # BERT模型的分词器
        self.bert_config = BertConfig.from_pretrained(self.bert_path)  # BERT模型的配置
        self.hidden_size = 768
