import torch
from config import Config
config = Config()

with open(config.orig_data_path,"r") as f:
    for line in f:
        line = line.strip()
        for