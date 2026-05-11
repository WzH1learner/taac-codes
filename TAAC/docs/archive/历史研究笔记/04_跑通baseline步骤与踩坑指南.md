# 跑通 Baseline 步骤与踩坑指南

## 官方平台流程（按照 shuoming.md）

### Step 1: 准备代码包

需要上传的文件结构：
```
train/
├── run.sh              # 必须！平台入口脚本
├── train.py            # 训练主入口
├── model.py            # PCVRHyFormer 模型
├── dataset.py          # 数据加载
├── trainer.py          # 训练循环
├── utils.py            # 工具函数
└── ns_groups.json      # NS 特征分组（可选）
```

**关键注意**：
- `run.sh` 是 **唯一入口**，平台启动任务时自动执行
- 不要遗漏任何 `.py` 文件，模型定义在 `model.py` 里
- 如果走 `group` tokenizer 模式，需要 `ns_groups.json`

### Step 2: 创建训练任务

1. 进入腾讯 AngelML 平台 → 模型训练模块
2. 点击「Create Training」
3. 填写：
   - Job Name: 任意名称
   - Job Description: 描述
4. 选择「本地上传」→ 上传整个 `train/` 目录（zip 或直接上传）
5. 点击「提交」

### Step 3: 平台自动注入的环境变量

任务启动后，平台会自动设置：

| 环境变量 | 平台自动提供 | 你的代码中使用 |
|---|---|---|
| `TRAIN_DATA_PATH` | ✅ 训练数据目录 | `os.environ.get('TRAIN_DATA_PATH')` |
| `TRAIN_CKPT_PATH` | ✅ 模型保存目录 | `os.environ.get('TRAIN_CKPT_PATH')` |
| `TRAIN_LOG_PATH` | ✅ 日志目录 | `os.environ.get('TRAIN_LOG_PATH')` |
| `TRAIN_TF_EVENTS_PATH` | ✅ TensorBoard | `os.environ.get('TRAIN_TF_EVENTS_PATH')` |

**你的代码里已经处理了**：
```python
args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
```

### Step 4: 数据与 schema.json 问题

**好消息**：
- 官方平台的正式数据集 **已经包含 schema.json**
- demo 数据（1000 条本地样本）缺失 schema.json 是已知问题，不影响平台运行
- 平台数据路径：`TRAIN_DATA_PATH/*.parquet` + `TRAIN_DATA_PATH/schema.json`

**所以答案是：按照 shuoming.md 的流程，在官方平台上传代码后可以直接跑通！**

### Step 5: 模型发布

训练完成后：
1. 进入训练任务 → 点击「Instances」
2. 进入实例页 → 点击「Output」
3. 选择要发布的 ckpt 文件（`global_stepXXX/model.pt`）
4. 点击「Publish」，填写模型名称和描述

### Step 6: 模型评估

发布模型后：
1. 模型管理页 → 选中模型 → 点击「Model Evaluation」
2. 或直接进「模型评估」页 → 「Create Evaluation」

需要上传 `eval/` 目录：
```
eval/
├── infer.py            # 必须！包含无参数的 main() 函数
├── dataset.py          # 与训练一致
├── model.py            # 与训练一致
└── prepare.sh          # 可选，安装依赖
```

评估环境变量（平台自动注入）：
| 环境变量 | 用途 |
|---|---|
| `MODEL_OUTPUT_PATH` | 你发布的 ckpt 目录 |
| `EVAL_DATA_PATH` | 测试数据集 |
| `EVAL_RESULT_PATH` | 输出 predictions.json |
| `EVAL_INFER_PATH` | 推理脚本目录 |

**predictions.json 格式**：
```json
{
    "predictions": {
        "用户ID字符串": 0.85,
        "用户ID字符串2": 0.32,
        ...
    }
}
```

**关键要求**：
- 键必须是测试集内的有效 user_id
- 不能缺失也不能多出无关 ID
- 值必须是 0~1 的浮点数（转化概率）

---

## 常见踩坑点

### 坑 1: run.sh 权限问题
如果平台报权限错误，确保 run.sh 有执行权限：
```bash
chmod +x run.sh
```

### 坑 2: 依赖缺失
如果代码依赖包平台没有，创建 `prepare.sh`：
```bash
#!/bin/bash
pip install some-package
```

### 坑 3: 路径问题
不要写死绝对路径，全部用环境变量：
```python
# ✅ 正确
data_dir = os.environ.get('TRAIN_DATA_PATH', './data')

# ❌ 错误
data_dir = '/home/user/data'
```

### 坑 4: schema.json 不匹配
训练和评估必须使用 **同一个 schema.json**
- 训练时 `trainer.py` 会自动把 schema.json 复制到 ckpt 目录
- 评估时 `infer.py` 会从 ckpt 目录读取 schema.json

### 坑 5: GPU 内存不足
如果报错 CUDA out of memory：
- 减小 `batch_size`（如 256 → 128 → 64）
- 减小 `d_model`（如 64 → 32）
- 减小 `seq_max_lens`（如 512 → 256）
- 开启 `emb_skip_threshold`（跳过超大 vocab 的 embedding）

### 坑 6: 检查点命名规范
保存路径必须满足：
```
global_step<数字>.<param>=<value>.<param2>=<value2>
```
如：`global_step1000.layer=2.head=4.hidden=64`

---

## 推荐的首次上传配置

### run.sh 内容（精简版）
```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --num_hyformer_blocks 2 \
    --d_model 64 \
    --batch_size 256 \
    --lr 1e-4 \
    --num_epochs 10 \
    --patience 5 \
    --device cuda \
    --num_workers 4 \
    "$@"
```

### 训练参数说明
| 参数 | 值 | 说明 |
|---|---|---|
| `batch_size` | 256 | 根据 GPU 内存调整 |
| `d_model` | 64 | 隐藏维度 |
| `num_hyformer_blocks` | 2 | 层数 |
| `num_queries` | 2 | 每个序列域生成 2 个 query |
| `lr` | 1e-4 | 稠密参数学习率 |
| `sparse_lr` | 0.05 | 稀疏参数学习率 |
| `patience` | 5 | 早停耐心值 |

---

## 快速验证清单

- [ ] 代码包包含 `run.sh` 且为入口
- [ ] `train.py` 从环境变量读取路径
- [ ] `model.py` + `dataset.py` + `trainer.py` + `utils.py` 完整
- [ ] `eval/infer.py` 包含无参数的 `main()` 函数
- [ ] 评估输出格式为 `{"predictions": {"user_id": probability}}`
- [ ] GPU 内存预估：模型参数量 × batch_size × 4 bytes

---
**整理时间**：2026-04-25
