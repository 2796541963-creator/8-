import torch
from torch.utils.data import DataLoader
from config import Config
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

config = Config()

# 处理原始数据，生成标签文档，返回基础处理数据
def load_raw_data():
    data = []
    fault_set = set()
    risk_grad_set = set()
    department_set = set()
    with open(config.orig_data_path,encoding='utf-8') as f:
            for lines in f:
                lines = lines.strip()
                if not lines:
                    continue
                text,fault,risk_grad,department = lines.split("\t")
                data.append((text,fault,risk_grad,department))
                fault_set.add(fault)
                risk_grad_set.add(risk_grad)
                department_set.add(department)
    # print(data[:20])
    # print(fault_set)
    # print(risk_grad_set)
    # print(department_set)
    fault_list = sorted(list(fault_set))
    id_to_fault = {idx:name for idx,name in enumerate(fault_list)}
    risk_grad_list = sorted(list(risk_grad_set))
    id_to_risk_grad  = {idx:name for idx,name in enumerate(risk_grad_list)}
    department_list = sorted(list(department_set))
    id_to_department  = {idx:name for idx,name in enumerate(department_list)}

    with open(config.fault_class_path,'w',encoding='utf-8') as f:
        for idx,name in id_to_fault.items():
            f.write(f"{idx}\t{name}\n")
    with open(config.risk_grad_class_path,'w',encoding='utf-8') as f:
        for idx,name in id_to_risk_grad.items():
            f.write(f"{idx}\t{name}\n")
    with open(config.department_class_path,'w',encoding='utf-8') as f:
        for idx,name in id_to_department.items():
            f.write(f"{idx}\t{name}\n")
    return data
# 进行id到标签，标签到id的映射
def id2class():
    fault_to_id ={}
    risk_grad_to_id ={}
    department_to_id ={}
    id_to_fault = {}
    id_to_risk_grad = {}
    id_to_department = {}
    with open(config.fault_class_path,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            parts  = line.split("\t")
            fault_to_id[parts[1]] = int(parts[0])
            id_to_fault[int(parts[0])] = parts[1]
    with open(config.risk_grad_class_path,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            parts  = line.split("\t")
            risk_grad_to_id[parts[1]] = int(parts[0])
            id_to_risk_grad[int(parts[0])] = parts[1]
    with open(config.department_class_path,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            parts  = line.split("\t")
            department_to_id[parts[1]] = int(parts[0])
            id_to_department[int(parts[0])] = parts[1]
    # print(fault_dict)
    # print(risk_grad_dict)
    # print(department_dict)
    return fault_to_id,id_to_fault,risk_grad_to_id,id_to_risk_grad,department_to_id,id_to_department
# 将标签映射为0.1.2....的数字
def dataset():
    data = load_raw_data()
    fault_to_id, id_to_fault, risk_grad_to_id, id_to_risk_grad, department_to_id, id_to_department= id2class()
    data = pd.DataFrame(data,columns=['text','fault','risk_grad','department'])
    data['fault'] = data['fault'].map(fault_to_id)
    data['risk_grad'] = data['risk_grad'].map(risk_grad_to_id)
    data['department'] = data['department'].map(department_to_id)
    print(data.head())
    return data
# 完成bert的特征编码
def collate_fn(batch):
    texts = [item[0] for item in batch]
    fault = [item[1] for item in batch]
    risk_grad = [item[2] for item in batch]
    department = [item[3] for item in batch]
    text_tokens = config.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=config.pad_size
    )
    token_ids_list = text_tokens["input_ids"]
    token_attention_mask = text_tokens["attention_mask"]

    input_ids = torch.tensor(token_ids_list,dtype=torch.long)
    attention_mask = torch.tensor(token_attention_mask,dtype=torch.long)
    fault = torch.tensor(fault,dtype=torch.long)
    risk_grad = torch.tensor(risk_grad,dtype=torch.long)
    department = torch.tensor(department,dtype=torch.long)
    return input_ids,attention_mask,fault,risk_grad,department

# 测试集，训练集，验证集拆分
def dataloader():
    np.random.seed(42)
    df= dataset()
    # print(df)
    df['combined_label'] = (
        df['fault'].astype(str) + '_' +
        df['risk_grad'].astype(str) + '_' +
        df['department'].astype(str)
    )
    # 第一层拆分训练+验证，测试集
    x_temp,x_test,y_temp,y_test = train_test_split(df.drop(['fault','risk_grad','department','combined_label'],
                                                axis=1),df[['fault','risk_grad','department','combined_label']],test_size=0.15,
                                                random_state=42,stratify=df['combined_label'])
    # 第二层拆分训练，测试集
    x_train, x_val, y_train, y_val = train_test_split(x_temp,y_temp,test_size=0.1765,
                                                        random_state=42,stratify=y_temp['combined_label'])


    train_dataset = list(zip(
        x_train['text'].tolist(),
        y_train['fault'].tolist(),
        y_train['risk_grad'].tolist(),
        y_train['department'].tolist()
    ))
    val_dataset = list(zip(
        x_val['text'].tolist(),
        y_val['fault'].tolist(),
        y_val['risk_grad'].tolist(),
        y_val['department'].tolist()
    ))
    test_dataset = list(zip(
        x_test['text'].tolist(),
        y_test['fault'].tolist(),
        y_test['risk_grad'].tolist(),
        y_test['department'].tolist()
    ))

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    dataloader()







