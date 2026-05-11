# TAAC2026 Demo 数据说明

本目录保存 TAAC2026 官方样例数据，用于本地理解字段结构、调试数据读取和验证特征工程逻辑。更完整的项目级说明见 [docs/2-参考资料/样例数据说明.md](../docs/2-参考资料/样例数据说明.md)。

## 数据概览

| 项目 | 内容 |
|---|---|
| 文件 | `demo_1000.parquet` |
| 行数 | 1,000 |
| 列数 | 120 |
| 文件大小 | 约 39 MB |
| 存储格式 | Parquet，flat column layout |

flat column layout 表示所有特征都是顶层列，不使用嵌套 struct。

## 字段类别

| 类别 | 数量 | 数据类型 | 说明 |
|---|---:|---|---|
| ID & Label | 5 | `int64` / `int32` | 核心 ID、标签和时间戳 |
| User Int Features | 46 | `int64` / `list<int64>` | 用户离散特征 |
| User Dense Features | 10 | `list<float>` | 用户连续向量特征 |
| Item Int Features | 14 | `int64` / `list<int64>` | 物品离散特征 |
| Domain Sequence Features | 45 | `list<int64>` | 4 个行为域序列特征 |

## ID 与标签列

| 列名 | 类型 |
|---|---|
| `user_id` | `int64` |
| `item_id` | `int64` |
| `label_type` | `int32` |
| `label_time` | `int64` |
| `timestamp` | `int64` |

当前训练代码中 `label_type == 2` 为正样本，其余为负样本。

## 用户特征

- `user_int_feats_{1,3,4,48-59,82,86,92-109}`：标量 `int64`，共 35 列。
- `user_int_feats_{15,60,62-66,80,89-91}`：数组 `list<int64>`，共 11 列。
- `user_dense_feats_{61-66,87,89-91}`：数组 `list<float>`，共 10 列。

当 `user_int_feats_{fid}` 与 `user_dense_feats_{fid}` 共享相同 fid 时，它们通常共同描述同一实体或信号。

## 物品特征

- `item_int_feats_{5-10,12-13,16,81,83-85}`：标量 `int64`，共 13 列。
- `item_int_feats_{11}`：数组 `list<int64>`，共 1 列。

## 序列特征

| 行为域 | 列名范围 | 列数 |
|---|---|---:|
| domain A | `domain_a_seq_{38-46}` | 9 |
| domain B | `domain_b_seq_{67-79,88}` | 14 |
| domain C | `domain_c_seq_{27-37,47}` | 12 |
| domain D | `domain_d_seq_{17-26}` | 10 |

## 读取示例

```python
import pandas as pd

df = pd.read_parquet("demo_1000.parquet")
print(df.shape)       # (1000, 120)
print(df.columns)
```

## 注意

demo 数据只用于结构理解。平台正式训练数据包含 `schema.json`，训练和 eval 都应以平台数据目录中的 schema 为准。
