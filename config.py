import torch
import os

class Config(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_path = "./data"
        self.train_path = self.data_path + "/train.txt"
        self.test_path = self.data_path + "/test.txt"
        self.val_path = self.data_path + "/val.txt"
        self.orig_data_path = self.data_path +"/manufacturing_repair_text_dataset_cn.txt"
