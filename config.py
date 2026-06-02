import torch
import os


class Config(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_path = "./data"
        self.train_path = self.data_path + "/train.txt"
        self.test_path = self.data_path + "/test.txt"
        self.dev_path = self.data_path + "/dev.txt"
        self.label2id_path = self.data_path + "/label2id.json"
        self.orig_data_path = (
            self.data_path + "/manufacturing_repair_text_dataset_cn.txt"
        )

        self.model_save_path = "./saved_models"
        self.bert_path = "./bert-base-chinese"
        
        self.num_classes = {
            "fault_type": 10,
            "risk_level": 4,
            "department": 9,
        }

        self.max_length = 55
        self.batch_size = 32
        self.learning_rate = 2e-5
        self.num_epochs = 2
