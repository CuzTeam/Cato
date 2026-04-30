# Cato

中文敏感词检测系统，基于 LR + TF-IDF + 词表特征工程。

## 检测类别

色情 · 反动 · 政治 · 脏话侮辱 · 道德伦理 · 身体伤害 · 偏见歧视

## 模型指标

| 指标 | 值 |
|------|-----|
| Accuracy | 97.6% |
| F1 macro | 0.976 |
| 误判率 | 2.3% |

## 快速开始

```bash
# 克隆（含词表子模块）
git clone --recurse-submodules https://github.com/<your-repo>/Cato.git
cd Cato

# 安装依赖
uv sync

# 训练 + 评估
uv run python main.py
```

训练完成后模型保存在 `datasets/model_v4/`，进入交互模式可输入文本实时打分。

## HTTP API

```bash
# 先训练一次模型，然后启动服务
uv run python http_server.py
# → 监听 http://0.0.0.0:8421
```

### 单条检测

```bash
curl -X POST http://localhost:8421/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "你好世界"}'
```

响应：

```json
{
  "score": 0.1001,
  "verdict": "正常",
  "notable_words": []
}
```

### 批量检测

```bash
curl -X POST http://localhost:8421/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["你好世界", "今天天气真好"]}'
```

响应：

```json
{
  "results": [
    {"text": "你好世界", "score": 0.1001, "verdict": "正常"},
    {"text": "今天天气真好", "score": 0.3415, "verdict": "正常"}
  ],
  "total": 2
}
```

### 健康检查

```bash
curl http://localhost:8421/health
# → {"status":"ok"}
```

## 评估

```bash
uv run python eval.py
```

输出全局指标 + 按 subject / 文本长度拆分的分类报告 + FP/FN 详细分析。

## 项目结构

```
Cato/
├── main.py              模型核心（训练/预测/评估）
├── http_server.py       FastAPI 服务
├── eval.py              诊断评估脚本
├── pyproject.toml       项目依赖
├── datasets/
│   ├── chinese_safe.jsonl   训练语料
│   └── model_v4/            模型文件（需训练生成）
├── challenges/
│   └── c1.jsonl             Challenge 测试集
└── words_repo/              敏感词表（git submodule）
    └── Vocabulary/          词库 .txt 文件
```

## 算法

- **分类器**: Logistic Regression (C=10.0, class_weight=balanced)
- **特征**: 词级 TF-IDF (1-2gram) + 字符级 TF-IDF (2-4gram) + 词表子串匹配特征 + 文本统计特征 + 正式度特征 + 短文本增强特征
- **词表**: 9400+ 敏感词，5 大类别，子串匹配覆盖变体词
- **阈值**: 0.60（≥0.60 判违规，0.35~0.60 判疑似，<0.35 判正常）

## 依赖

- Python ≥ 3.11
- jieba
- scikit-learn
- fastapi + uvicorn（API 服务）

## 数据来源

- **训练语料**: [ChineseSafe](https://huggingface.co/datasets/SUSTech/ChineseSafe) by SUSTech — CC-BY-NC-4.0
- **敏感词表**: [Sensitive-lexicon](https://github.com/konsheng/Sensitive-lexicon) by konsheng
