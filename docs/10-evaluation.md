# 10 · 评测与回归

> 一句话：改 Prompt 或切分后必须跑全集；简历数字只能来自 `eval/runner` 与 [`MEASURED.md`](../eval/MEASURED.md)。完整口径、消融与路径表见 [14 · 测评报告](14-evaluation-report.md)。

## 本章目录

1. [命令与数据集](#命令与数据集)
2. [指标怎么算](#指标怎么算)
3. [已测量](#已测量须与代码一起引用)
4. [可以写 / 不能写](#简历可以写--不能写)
5. [口径陷阱](#口径陷阱面试主动说)
6. [CI](#ci-建议)

对应：`eval/runner.py`、`eval/metrics.py`、`eval/dataset/`。

## 命令与数据集

```bash
python -m eval.runner --task all --llm
python -m eval.runner --task retrieval          # CI；装了 .[rerank] 且能加载权重时含 Cross-Encoder
python -m eval.runner --task multihop --llm     # 20 条多跳切片
python -m eval.runner --task intent_holdout --llm  # 43 条意图留出集
EVAL_JUDGE_MODEL=cn:glm-5.3 python -m eval.runner --task generation --llm  # 独立裁判
python -m eval.runner --task intent --llm
python -m eval.runner --task generation --llm
python -m eval.runner --task e2e --llm
```

报告：`eval/reports/*.json`（git 忽略）。

| 文件 | 用途 | 条数 |
|---|---|---|
| `eval/dataset/intent_100.jsonl` | 14 类意图（训练/调参） | 100 |
| `eval/dataset/intent_holdout.jsonl` | 意图留出集（未参与调参） | 43 |
| `eval/dataset/rag_eval.jsonl` | 政策检索 + 关键词 | 65 |
| `eval/dataset/rag_eval_multihop.jsonl` | 多跳/跨条款（每条 ≥2 篇相关文档） | 20 |
| `eval/dataset/e2e_eval.jsonl` | 注入/订单/政策 | 16 |

## 指标怎么算

文档级：chunk 的 `document_id` 去掉 `_{index}` 得到 stem，与 `relevant_doc_ids` 比。

E2E 的分阶段延迟只统计实际执行过的阶段（`*_ms > 0`），并同时输出 `intent_n / retrieval_n / llm_n / tool_n` 样本数；未检索的 case 不再把 P50 拉到 0。

### 代码摘要

```python
# eval/metrics.py
def precision_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    top = retrieved[:k]
    return sum(item in relevant for item in top) / len(top)

def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    return sum(item in relevant for item in retrieved[:k]) / len(relevant)

def mrr(relevant: set[str], retrieved: list[str]) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0
```

意图：`accuracy = 命中数 / 100`。生产评测跑 `intent_v3` + overlay。  
生成：幻觉 = 忠实度裁判 `supported=false`。相对下降 = `(naive - checked) / naive`。裁判与生成器同模型，**偏乐观**；关键词命中更独立。

检索评测在未加载 Cross-Encoder 时**不会**把词面 overlap 标成 Cross-Encoder。

## 已测量（须与代码一起引用）

**意图**（`cn:deepseek-v4.1-flash`，100 条）

| 版本 | 准确率 |
|---|---|
| 关键词规则 | 50% |
| `intent_v1`（无覆盖，历史） | 88% |
| `intent_v2`（无覆盖，历史） | 87% |
| 仅歧义覆盖层 | **100%** |
| `intent_v3` + 覆盖（生产，训练集） | **100%** |

**意图留出集**（43 条，未参与调参）：规则 39.5%；仅覆盖层 39.5%（25 条未覆盖）；`intent_v3`+覆盖 **95.3%（41/43）**。两条错误：「拆封的耳机能退吗」通用/具体售后边界、「头疼吃什么药」医疗越界。

**检索**（65 条，39 篇；**已加载 `bge-small-zh-v1.5` + Cross-Encoder**）

| 系统 | Precision@1 | Recall@3 | MRR |
|---|---|---|---|
| BM25 | 80.0% | 90.0% | 0.86 |
| 仅 `bge-small-zh-v1.5` | 81.5% | 94.6% | 0.88 |
| BM25 + small + RRF | 83.1% | 94.6% | 0.89 |
| Hybrid + Cross-Encoder | **89.2%** | **96.2%** | **0.93** |

本轮检索数字与 2026-08-22 基线有小幅差异，主要来自 `knowledge/` 文档更新。CE 相对 BM25：P@1 +9.2、R@3 +6.2 个百分点。

**生成 / 幻觉**（65 条）

同模型裁判（生成器 = 裁判 = `flash`）：

| | Faithfulness | 幻觉率 | 关键词命中 |
|---|---|---|---|
| 无检索 | 32.3% | 67.7% | 36.9% |
| RAG | 100% | 0% | 95.4% |
| RAG + LLM 自纠错 | **100%** | **0%**（&lt;4%） | 95.4% |

独立裁判（`cn:glm-5.3`，判同一批答案）：

| | Faithfulness | 幻觉率 |
|---|---|---|
| 无检索 | 40.0% | 60.0% |
| RAG | 84.6% | 15.4% |
| RAG + LLM 自纠错 | **89.2%** | **10.8%** |

两裁判在 RAG 答案上的支持判定一致率 **84.6%**。同模型把 RAG 从 84.6% 抬到 100%，说明自评偏乐观约 15 个百分点；自纠错在独立裁判下是 +4.6pp，**不是 0 幻觉**。无检索 Faithfulness 随裁判模型变化，不能跨模型比较绝对值。

**多跳**（20 条，2026-11-19 11:20Z；每条 ≥2 篇相关文档）

| 系统 | P@1 | R@3 | R@5 | Hit@3 | MRR |
|---|---|---|---|---|---|
| BM25 | 90.0% | 75.8% | 83.3% | 100% | 0.95 |
| 仅 `bge-small-zh-v1.5` | 95.0% | 68.3% | 75.8% | 100% | 0.975 |
| BM25 + small + RRF | 85.0% | 78.3% | 83.3% | 100% | 0.925 |
| Hybrid + Cross-Encoder | 90.0% | 76.7% | 80.8% | 100% | 0.95 |

生成：无检索 35% / 65% / 25%（Faithfulness / 幻觉 / 关键词）→ RAG **100% / 0% / 95%** → 自纠错同 RAG。所有系统都能在 top-3 命中至少一篇，但多文档覆盖 R@5 只有 75.8%–83.3%；多跳收益主要在生成，不在召回完整性。

**E2E**（16 条，2026-11-19 07:57Z）：拒答 / 工具路径 100%；关键词 **11/11（100%）**；P50 **8.08s** / P95 **9.21s**。

## 简历可以写 / 不能写

**可以：** 规则 50% → intent_v3+覆盖训练集 100%、留出集 95.3%；仅 small P@1 81.5%、R@3 94.6%；Hybrid+CE P@1 89.2%、R@3 96.2%；BM25 80.0% / 90% / 0.86；同模型 RAG+自纠错幻觉 0%、独立裁判 10.8%（一致率 84.6%）；多跳 20 条 R@5 80.8%、Hit@3 100%。

**不能：** 把训练集 100% 当泛化 100%；把同模型 0% 幻觉当独立结论；把无检索 Faithfulness 跨模型比较；把稠密路写成 bge-m3；把 RRF 说成本集上显著优于仅 small。

## 口径陷阱（面试主动说）

1. 生产意图 = LLM JSON + 覆盖层；覆盖针对稳定中文歧义，不是按 query id 写死。
2. Faithfulness 同模型裁判偏乐观。
3. 稠密路是 `bge-small-zh-v1.5`，不是 bge-m3；本集上 RRF 没有超过仅 small。
4. pytest 不打真实 LLM。

## CI 建议

- PR：`pytest -q` + `python -m eval.runner --task retrieval`
- Prompt 变更：`--task intent --llm`，看 `mismatches`
- 知识库变更：先改 `rag_eval.jsonl` 金标
- 配上 Cross-Encoder：报告里必须出现 `hybrid+cross-encoder`

---

完整测评报告：[14](14-evaluation-report.md)。

上一章 [09](09-security.md) · 下一章 [11 · 部署](11-deployment.md)
