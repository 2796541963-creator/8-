# 设备报修文本分类项目协作说明

## 1. 项目目录作用

本项目用于构建设备报修文本故障分类与停机风险预警系统。系统目标是输入一条制造业设备报修文本，输出故障大类、停机风险等级、推荐处理部门、置信度和是否需要人工复核。

### 1.1 `data/`

`data/` 用于保存项目数据。当前已有原始数据集 `manufacturing_repair_text_dataset_cn.txt`，该文件为 UTF-8 编码、Tab 分隔文本，共 220000 行，每行包含 4 个字段：

```text
text    fault_category    risk_level    department
```

建议后续将数据目录拆分为：

```text
data/
  raw/          原始数据，不做覆盖修改
  processed/    清洗、去泄漏、切分后的训练数据
  reports/      EDA 报告、类别分布图、评估报告
```

`raw/` 中的数据只读保存，所有清洗和切分结果都写入 `processed/`，避免训练脚本直接污染原始语料。

### 1.2 `docs/`

`docs/` 用于保存项目背景、业务方案、标签体系、公开数据源和答辩材料。当前文档已经覆盖项目实施方案、文本分类体系、公开数据源清单和相关回复整理。

后续如新增模型设计、接口设计、训练报告、部署说明，也应放入 `docs/`，保证业务说明和工程实现可以互相追溯。

### 1.3 `model/` 本地模型目录

`model/` 用于保存预训练模型、训练后模型和导出模型，但本 PR 不提交该目录。原因是 MacBERT 权重文件较大，普通 Git 仓库不适合直接保存模型权重。

训练或推理前，可在本地将中文 MacBERT 下载到 `model/chinese-macbert-base/`。建议只在本地保留 `pytorch_model.bin`、`tokenizer.json`、`vocab.txt`、`config.json` 等文件，或者后续改用 Git LFS、Hugging Face、ModelScope、对象存储管理权重。

建议后续继续拆分：

```text
model/
  chinese-macbert-base/     原始预训练 MacBERT
  baseline/                 TF-IDF + LinearSVC 等传统模型
  macbert_multitask/         微调后的多任务分类模型
  exported/                 用于 Flask 推理服务加载的导出模型
```

### 1.4 `scripts/`

`scripts/` 用于保存可重复执行的数据处理、训练、评估和推理脚本。脚本应尽量通过命令行参数接收输入输出路径，不在代码中写死绝对路径。

脚本职责应保持单一：EDA 脚本只做探索分析，清洗脚本只做清洗，切分脚本只做 train/valid/test 切分，训练脚本只做模型训练。

### 1.5 `app/`

`app/` 建议用于保存 Flask 应用代码，当前目录尚未创建。Flask 应同时服务 Web 前端展示和小程序 API。

建议结构：

```text
app/
  __init__.py
  routes.py
  services/
    predictor.py
    schema.py
  templates/
    index.html
    dashboard.html
  static/
    css/
    js/
```

`routes.py` 负责 Flask 路由，`predictor.py` 负责模型加载和推理，`schema.py` 负责请求和响应格式校验。

### 1.6 `configs/`

`configs/` 建议用于保存训练配置、标签配置、接口配置和模型路径配置，当前目录尚未创建。

建议结构：

```text
configs/
  labels.json
  train_config.yaml
  api_config.yaml
```

`labels.json` 保存故障大类、风险等级、推荐部门的 label2id 和 id2label。`train_config.yaml` 保存模型名、最大长度、学习率、batch size、epoch、loss 权重等训练参数。

### 1.7 `run.py`

`run.py` 建议作为 Flask 应用启动入口，负责读取配置、创建 Flask app、启动 Web 服务。

运行方式建议为：

```powershell
python run.py
```

启动后访问：

```text
http://127.0.0.1:5000
```

## 2. MacBERT 架构说明

当前下载的模型为 `hfl/chinese-macbert-base`。它来自论文《Revisiting Pre-trained Models for Chinese Natural Language Processing》，模型地址为：

```text
https://huggingface.co/hfl/chinese-macbert-base
```

MacBERT 的主体神经网络架构与 BERT 基本一致，本地 `config.json` 显示其核心参数如下：

```text
model_type: bert
architectures: BertForMaskedLM
num_hidden_layers: 12
hidden_size: 768
num_attention_heads: 12
intermediate_size: 3072
max_position_embeddings: 512
vocab_size: 21128
hidden_act: gelu
hidden_dropout_prob: 0.1
attention_probs_dropout_prob: 0.1
```

可以把 MacBERT 理解为“BERT-base 的中文增强预训练版本”。它并不是重新发明 Transformer 结构，而是在预训练任务上做了改进。

普通 BERT 的 MLM 会把词替换为 `[MASK]`，但 `[MASK]` 在真实微调和推理阶段不会出现，这会造成预训练和下游任务之间的差异。MacBERT 将 MLM 改造成 MLM as Correction，即用相似词替换原词，让模型学习从“相似但不准确的词”纠正回原词。

例如普通 MLM 更像这样：

```text
设备 [MASK] 后无法启动
```

MacBERT 的思路更像这样：

```text
设备 运行 后无法启动
```

模型需要判断上下文里“运行”并不合适，并恢复更准确的语义。这种预训练方式更接近真实文本中的错词、近义词、口语化表达和工业报修中的不规范描述。

MacBERT 还结合了 Whole Word Masking、N-gram Masking 和 Sentence Order Prediction。对于中文设备报修文本来说，Whole Word Masking 尤其有价值，因为“光电传感器”“安全光栅”“控制柜”“变频器”这类术语应尽量作为完整语义单元被理解。

本项目推荐使用 MacBERT 的 `[CLS]` 向量作为句子级表示，再接三个分类头，分别预测：

```text
fault_category    故障大类
risk_level        停机风险等级
department        推荐处理部门
```

多任务损失建议为：

```text
L = L_fault + 1.5 * L_risk + L_department
```

其中 `risk_level` 权重更高，因为工业场景中 P0/P1 高风险工单漏判的代价更大。

## 3. 多 Agent 协作分工

如果使用多 agent 协作，建议按以下角色拆分。

### 3.1 Data Agent

负责数据探索、清洗、去标签泄漏、数据切分和数据报告。

主要产物：

```text
scripts/dataset_eda.py
scripts/dataset_clean.py
scripts/dataset_split.py
data/processed/train.tsv
data/processed/val.tsv
data/processed/test.tsv
data/reports/eda_summary.md
```

### 3.2 Model Agent

负责 baseline、MacBERT 多任务微调、模型评估和模型导出。

主要产物：

```text
scripts/train_baseline.py
scripts/train_macbert_multitask.py
scripts/evaluate_model.py
scripts/export_model.py
model/baseline/
model/macbert_multitask/
model/exported/
```

### 3.3 Backend Agent

负责 Flask 后端、模型加载服务、接口契约和小程序 API 预留。

主要产物：

```text
app/routes.py
app/services/predictor.py
app/services/schema.py
run.py
configs/api_config.yaml
```

### 3.4 Frontend Agent

负责 Flask 前端展示页，包括文本输入、预测结果展示、风险颜色标记、Dashboard 和历史记录页面。

主要产物：

```text
app/templates/index.html
app/templates/dashboard.html
app/static/css/
app/static/js/
```

## 4. `scripts/` 文件方案

### 4.1 `dataset_eda.py`

用途：对原始数据做探索性分析。

输入：

```text
data/manufacturing_repair_text_dataset_cn.txt
```

输出：

```text
data/reports/eda_report.md
data/reports/category_distribution.png
data/reports/risk_distribution.png
data/reports/department_distribution.png
```

核心检查项：

```text
总行数
空值数量
列数异常数量
重复文本数量
故障大类分布
风险等级分布
推荐部门分布
文本长度分布
类别与部门对应关系
```

### 4.2 `dataset_clean.py`

用途：清洗数据并减少标签泄漏。

输入：

```text
data/manufacturing_repair_text_dataset_cn.txt
```

输出：

```text
data/processed/cleaned_dataset.tsv
```

重点处理：

```text
去重
去空值
统一标点和空格
移除明显泄漏标签
记录清洗前后样本数量
```

需要重点移除或替换的泄漏短语：

```text
风险等级建议为P0
风险等级建议为P1
风险等级建议为P2
风险等级建议为P3
建议按P0工单处理
建议按P1工单处理
建议按P2工单处理
建议按P3工单处理
请尽快安排机械维修
请尽快安排电气维修
请尽快安排自动化工程师
请安全/EHS确认
请质量/工艺/设备确认
```

### 4.3 `dataset_split.py`

用途：将清洗后的数据切分为训练集、验证集和测试集。

输入：

```text
data/processed/cleaned_dataset.tsv
```

输出：

```text
data/processed/train.tsv
data/processed/val.tsv
data/processed/test.tsv
configs/labels.json
```

建议切分比例：

```text
train: 80%
valid: 10%
test: 10%
```

切分方式建议按 `fault_category + risk_level` 组合字段进行分层，避免某些风险等级在验证集或测试集中分布过少。

### 4.4 `train_baseline.py`

用途：训练传统机器学习 baseline。

推荐方案：

```text
TF-IDF + LinearSVC
```

训练任务：

```text
fault_category 单任务分类
risk_level 单任务分类
department 单任务分类
```

baseline 的作用不是作为最终模型，而是给 MacBERT 微调提供对照基准。如果 MacBERT 比 baseline 只高一点，说明数据可能存在模板过强或标签泄漏问题。

### 4.5 `train_macbert_multitask.py`

用途：训练 MacBERT 多任务分类模型。

输入：

```text
data/processed/train.tsv
data/processed/val.tsv
model/chinese-macbert-base/
configs/train_config.yaml
configs/labels.json
```

输出：

```text
model/macbert_multitask/
```

推荐训练参数：

```text
max_length: 128
batch_size: 16 或 32
learning_rate: 2e-5
epochs: 3 到 5
warmup_ratio: 0.1
weight_decay: 0.01
```

模型结构：

```text
MacBERT Encoder
  -> [CLS]
    -> fault_category_head
    -> risk_level_head
    -> department_head
```

### 4.6 `evaluate_model.py`

用途：统一评估 baseline 和 MacBERT 模型。

重点指标：

```text
Accuracy
Macro-F1
Weighted-F1
P0 Recall
P1 Recall
P0/P1 Combined Recall
Confusion Matrix
```

工业场景下不能只看整体准确率。P0/P1 的召回率更关键，因为高风险工单漏判的业务损失更大。

### 4.7 `export_model.py`

用途：导出 Flask 推理服务可加载的模型文件。

输出：

```text
model/exported/
```

导出内容：

```text
模型权重
tokenizer
label mappings
推理配置
```

后续可选导出 ONNX 或动态量化模型，用于降低 CPU 推理延迟。

### 4.8 `predict_cli.py`

用途：提供命令行预测能力，便于在 Flask 接入前验证模型。

示例：

```powershell
python scripts/predict_cli.py --text "PLC报警E102，设备无法复位"
```

输出示例：

```json
{
  "fault_category": "控制系统故障",
  "risk_level": "P0",
  "department": "自动化工程师",
  "need_human_review": true
}
```

## 5. Flask 与小程序接口方案

Flask 后端应同时服务 Web 前端和小程序接口。

推荐接口：

```text
GET  /
GET  /dashboard
GET  /api/v1/health
GET  /api/v1/labels
POST /api/v1/predict
POST /api/v1/predict/batch
POST /api/v1/feedback
```

### 5.1 单条预测接口

请求：

```json
{
  "text": "PLC报警E102，设备无法复位",
  "source": "wechat_miniapp",
  "user_id": "optional",
  "device_id": "optional"
}
```

响应：

```json
{
  "success": true,
  "data": {
    "fault_category": "控制系统故障",
    "risk_level": "P0",
    "department": "自动化工程师",
    "confidence": {
      "fault_category": 0.96,
      "risk_level": 0.91,
      "department": 0.93
    },
    "need_human_review": true,
    "suggestion": "高风险工单，建议优先派单处理"
  }
}
```

### 5.2 人工反馈接口

请求：

```json
{
  "text": "PLC报警E102，设备无法复位",
  "predicted": {
    "fault_category": "控制系统故障",
    "risk_level": "P0",
    "department": "自动化工程师"
  },
  "corrected": {
    "fault_category": "控制系统故障",
    "risk_level": "P0",
    "department": "自动化工程师"
  },
  "reviewer": "operator"
}
```

反馈数据应保存为后续主动学习和 hard examples 重训数据。

## 6. 推荐执行顺序

第一阶段：数据层

```powershell
python scripts/dataset_eda.py --input data/manufacturing_repair_text_dataset_cn.txt
python scripts/dataset_clean.py --input data/manufacturing_repair_text_dataset_cn.txt --output data/processed/cleaned_dataset.tsv
python scripts/dataset_split.py --input data/processed/cleaned_dataset.tsv --output-dir data/processed
```

第二阶段：模型层

```powershell
python scripts/train_baseline.py --train data/processed/train.tsv --valid data/processed/val.tsv
python scripts/train_macbert_multitask.py --config configs/train_config.yaml
python scripts/evaluate_model.py --model_dir model/macbert_multitask --test data/processed/test.tsv
```

第三阶段：服务层

```powershell
python run.py
```

访问：

```text
http://127.0.0.1:5000
```

第四阶段：小程序接入

小程序只需要对接 `/api/v1/predict`、`/api/v1/labels` 和 `/api/v1/feedback` 三个接口即可完成基础闭环。

## 7. 开发注意事项

1. 中文文件统一使用 UTF-8 编码。
2. 不要直接修改原始数据文件，清洗结果写入 `data/processed/`。
3. 训练前必须先处理标签泄漏，否则模型会学到文本中的答案提示。
4. 模型评估不能只看 Accuracy，必须重点看 Macro-F1 和 P0/P1 Recall。
5. Flask 接口返回结构必须稳定，方便小程序调用。
6. 大模型如果后续接入，只用于低置信度、高风险和冲突样本精筛，不替代小模型主分类器。
7. 每个脚本都应支持命令行参数，避免路径写死。
8. 每次训练应保存配置、标签映射、模型指标和随机种子，保证结果可复现。
