import pandas
import torch
from config import Config
import json
import pandas as pd

config = Config()


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
            fault_to_id[parts[1]] = parts[0]
            id_to_fault[parts[0]] = parts[0]
    with open(config.risk_grad_class_path,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            parts  = line.split("\t")
            risk_grad_to_id[parts[1]] = parts[0]
            id_to_risk_grad[parts[0]] = parts[1]
    with open(config.department_class_path,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            parts  = line.split("\t")
            department_to_id[parts[1]] = parts[0]
            id_to_department[parts[0]] = parts[1]
    # print(fault_dict)
    # print(risk_grad_dict)
    # print(department_dict)
    return fault_to_id,id_to_fault,risk_grad_to_id,id_to_risk_grad,department_to_id,id_to_department

def dataset():
    fault_to_id, id_to_fault, risk_grad_to_id, id_to_risk_grad, department_to_id, id_to_department= id2class()
    data = load_raw_data()
    data = pd.DataFrame(data,columns=['text','fault','risk_grad','department'])
    data['fault'] = data['fault'].map(fault_to_id)
    data['risk_grad'] = data['risk_grad'].map(risk_grad_to_id)
    data['department'] = data['department'].map(department_to_id)
    print(data.head())
    return data

def collate_fn(batch):
    texts = [item[0] for item in batch]
    fault = [item[1] for item in batch]
    risk_grad = [item[2] for item in batch]
    department = [item[3] for item in batch]
    text_tokens = config.tokenizer.batch_encode_plus(texts,padding=True)
    token_ids_list = text_tokens["input_ids"]
    token_attention_mask = text_tokens["attention_mask"]

    input_ids = torch.tensor(token_ids_list,dtype=torch.long)
    attention_mask = torch.tensor(token_attention_mask,dtype=torch.long)
    fault = torch.tensor(fault,dtype=torch.long)
    risk_grad = torch.tensor(risk_grad,dtype=torch.long)
    department = torch.tensor(department,dtype=torch.long)
    return input_ids,attention_mask,fault,risk_grad,department







