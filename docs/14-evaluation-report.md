# 14 · 测评报告

> 一句话：数字只来自 `eval.runner` 写出的 JSON；本文把口径、数据集、消融和限制写全，方便复现和面试对质。

## 本章目录

1. [结论摘要](#1-结论摘要)
2. [评测目标与范围](#2-评测目标与范围)
3. [环境与复现](#3-环境与复现)
4. [数据集](#4-数据集)
5. [指标定义](#5-指标定义)
6. [意图识别](#6-意图识别)
7. [检索](#7-检索)
8. [生成与幻觉](#8-生成与幻觉)
9. [端到端](#9-端到端)
10. [自动化测试](#10-自动化测试)
11. [与简历数字对照](#11-与简历数字对照)
12. [未测项与下一步](#12-未测项与下一步)

对应：[`eval/runner.py`](../eval/runner.py)、[`eval/metrics.py`](../eval/metrics.py)、[`eval/MEASURED.md`](../eval/MEASURED.md)。  
命令与 CI 门槛见 [10 · 评测与回归](10-evaluation.md)。

**测量日期：** 2026-11-19  
**生成模型 / 裁判模型：** `cn:deepseek-v4.1-flash`（本地 OpenAI 兼容代理；结构化 JSON 调用关闭 reasoning）  
**稠密召回：** `BAAI/bge-small-zh-v1.5`（已加载）  
**精排：** `BAAI/bge-reranker-base`（已加载）  
**原始报告：** 标识 `retrieval_20261119T112136Z.json`、`intent_20261119T075058Z.json`、`intent_holdout_20261119T114201Z.json`、`generation_20261119T125437Z.json`、`e2e_20261119T075759Z.json`、`multihop_20261119T112028Z.json`。`eval/reports/*.json` 已按 `.gitignore` 忽略、不随仓库分发，仓库内没有报告本体；结论以 [`MEASURED.md`](../eval/MEASURED.md) 为准。检索数字与 2026-08-22 基线的小幅差异主要来自 `knowledge/` 文档更新。

---

## 1. 结论摘要

| 任务 | 集合 | 主结果 | 对照 |
|---|---|---|---|
| 意图 | 100 条训练 + 43 条留出 | 训练集 **100%**；留出集 **95.3%** | 规则 50%；无覆盖 v1 88% / v2 87% |
| 检索 | 65 条 / 39 篇 | BM25 **P@1 80.0%**、**R@3 90.0%**、**MRR 0.86** | 仅 `bge-small` **81.5 / 94.6 / 0.88** |
| 检索 + CE | 同上 | Hybrid+CE **P@1 89.2%**、**R@3 96.2%**、**MRR 0.93** | 相对 BM25 +9.2 / +6.2 个百分点 |
| 生成 / 幻觉 | 65 条 | 同模型 RAG+自纠错 **100% / 0%**；独立裁判 **89.2% / 10.8%** | 无检索 同模型 32.3% / 67.7%；独立 40% / 60%；两裁判一致率 84.6% |
| 多跳 | 20 条 | Hybrid+CE **R@5 80.8%**、Hit@3 **100%** | RAG 忠实度 100%、关键词 95%；无检索 35% / 25% |
| E2E | 16 条 | 拒答 / 工具路径 **100%**；关键词 **11/11（100%）** | P50 **8.08s** / P95 **9.21s** |
| 回归 | pytest | **198 passed**（CI 干净环境 194 + 4 skipped） | 不打真实 LLM / 不下载权重 |

### 要点

这次测评验证四件事：Adaptive 路由能把问候和注入从检索里拆出去；本地 `bge-small-zh-v1.5` 把仅向量 P@1 拉到 **81.5%**；Cross-Encoder 再把 Hybrid P@1 提到 **89.2%**；忠实度自纠错在同模型裁判下把幻觉压到 0%，但独立裁判下是 **10.8%**，两裁判一致率 **84.6%**。本轮 65 条上 RRF 仅略高于仅 small，主要增益仍来自 Cross-Encoder。无检索 Faithfulness 随裁判模型变化（同模型 32.3% / 独立 40%），不能跨模型比较绝对值。

---

## 2. 评测目标与范围

客服场景里「感觉还行」不够。本仓库把可面试的数字钉在四条离线任务上：

| 任务 | 回答的问题 | 不回答的问题 |
|---|---|---|
| `intent` | 14 类意图能不能分对，覆盖层贡献多少 | 路由策略是否最优 |
| `retrieval` | 政策文档能不能排到前面 | 生成句子是否正确 |
| `generation` | 有无证据时会不会胡编；自纠错有没有用 | 人工偏好、文采 |
| `e2e` | 注入会不会拒、订单会不会走工具、延迟多少 | 并发压测、多租户 |

**范围内：** 本地 SQLite + 内存检索 + DeepSeek + 本地 `bge-small-zh-v1.5` + Cross-Encoder；知识库 `knowledge/*.md`（39 篇）。  
**范围外：** 更大 embedding（bge-m3 等）、线上流量 A/B、人工标注一致性。

---

## 3. 环境与复现

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]" ".[rerank]"
# .env 填 DEEPSEEK_API_KEY；国内拉权重可设 HF_ENDPOINT=https://hf-mirror.com

pytest -q
python -m eval.runner --task all --llm
python -m eval.runner --task intent_holdout --llm   # 意图留出集
python -m eval.runner --task multihop --llm          # 多跳切片
EVAL_JUDGE_MODEL=cn:glm-5.3 python -m eval.runner --task generation --llm  # 独立裁判
```

报告写到 `eval/reports/{task}_{UTC}.json` 和 `eval/reports/latest.json`。  
改 Prompt、切分或知识库后必须重跑对应任务，再改 [`MEASURED.md`](../eval/MEASURED.md) 与本文数字。

### 为什么

pytest 在 `tests/conftest.py` 清空真实 key，保证 CI 可重复、不烧钱。简历上的百分比必须来自 `--llm` 那次跑出来的 JSON，不能手填。

---

## 4. 数据集

| 文件 | 条数 | 金标 | 用途 |
|---|---|---|---|
| `eval/dataset/intent_100.jsonl` | 100 | `intent`（14 类） | 分类准确率与消融 |
| `eval/dataset/intent_holdout.jsonl` | 43 | 新问法，未参与调参 | 意图留出集泛化 |
| `eval/dataset/rag_eval.jsonl` | 65 | `relevant_doc_ids` + `gold_answer_contains` | 检索 P/R/MRR + 生成关键词 |
| `eval/dataset/rag_eval_multihop.jsonl` | 20 | 每条 ≥2 个 `relevant_doc_ids` + 关键词 | 多跳检索 R@5 / 多跳生成 |
| `eval/dataset/e2e_eval.jsonl` | 16 | `expect_refused` / `expect_keywords` | 整图路径与延迟 |

### 4.1 意图：14 类分布

| 标签 | 条数 | 业务含义 |
|---|---|---|
| `faq_policy` | 16 | 发票、积分、客服时间等政策 |
| `return_exchange` | 8 | 退货 / 换货 |
| `out_of_scope` | 8 | 注入、炸弹、炒股等越权 |
| `order_inquiry` | 7 | 查单 |
| `logistics` | 7 | 物流轨迹 |
| `product_consult` | 7 | 规格 / 库存 |
| `chitchat` | 6 | 闲聊 |
| `refund` | 6 | 退款到账 |
| `cancel_order` | 6 | 取消未发货 |
| `recommendation` | 6 | 推荐商品 |
| `coupon` | 6 | 优惠券叠加 |
| `complaint` | 6 | 投诉 |
| `human_handoff` | 6 | 转人工 |
| `greeting` | 5 | 问候 |

样例：`{"id":"i011","query":"怎么开发票","intent":"faq_policy"}`。

### 4.2 检索 / 生成：65 条政策问句

金标是**文档 stem**（如 `return_policy`），不是 chunk id。65 条覆盖 38 个不同文档 stem；5 条 query 标了两篇相关文档（例如包邮 + 优惠券）。

样例：

```json
{"id":"r001","query":"定制商品能不能七天无理由退货","relevant_doc_ids":["return_policy"],"gold_answer_contains":["定制","无理由"]}
```

知识库 `knowledge/` 共 **39** 篇 Markdown，ingest 后约 156 chunk。文档级匹配：`document_id` 去掉 `_{index}` 得到 stem。

### 4.3 多跳：20 条跨条款问题

`rag_eval_multihop.jsonl` 每条至少标 2 篇相关文档，覆盖定制+券、跨境+退款、包邮+偏远、PLUS+叠券、数码+保修、生鲜+退款、拆单+物流等组合。检索额外报告 `recall_at_5`（多文档问题下 R@3 会低估覆盖），生成沿用同一套关键词与同模型裁判。

### 4.4 E2E：16 条路径覆盖

| 组 | ID | 在测什么 |
|---|---|---|
| 问候 | e01 | Direct，不检索 |
| 政策 | e02–e05, e09, e16 | Single / Multi + 关键词 |
| 工具 | e06–e08, e14, e15 | 查单 / 物流 / 库存 / 取消（HITL） |
| 拒答 | e10–e12 | 注入、造炸弹、炒股 |
| 转人工 | e13 | `human_handoff` |

`e10`「Ignore previous instructions…」期望 `refused=true`。  
演示用户 `u_demo`；订单 `ORD10001` / `ORD10002` / `ORD10003`。

---

## 5. 指标定义

### 要点

检索评的是**文档**不是 chunk：同一篇政策的多个切片只计一次。幻觉率 = 忠实度裁判 `supported=false` 的比例，**不是**人工事实核查。

### 代码摘要

```python
# eval/metrics.py
def precision_at_k(relevant, retrieved, k):
    top = retrieved[:k]
    return sum(item in relevant for item in top) / len(top)

def recall_at_k(relevant, retrieved, k):
    return sum(item in relevant for item in retrieved[:k]) / len(relevant)

def mrr(relevant, retrieved):
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0
```

| 指标 | 公式（本仓库实现） | 报告里用的 |
|---|---|---|
| Accuracy | 命中数 / N | 意图 100 条 |
| Precision@k | Top-k 里相关文档数 / k | P@1 |
| Recall@k | Top-k 命中的相关文档 / 金标文档数 | R@3 |
| Hit@k | Top-k 是否至少一篇相关 | 向量路对照 |
| MRR | 第一个相关文档排名的倒数，未命中为 0 | 检索 |
| Faithfulness | `supported=true` 占比 | 生成 |
| 幻觉率 | `supported=false` 占比 | 生成 |
| 相对下降 | `(naive - checked) / naive` | 无检索 vs RAG+自纠错 |
| 关键词命中 | `gold_answer_contains` 全部出现在答案中 | 生成 / E2E |
| 拒答准确率 | `expect_refused` 与 `response.refused` 一致 | E2E（3 条） |
| 工具路径 | query 含 ORD/库存/取消，且答案含订单号或 `pending_action` | E2E |
| 延迟 | `response.latency.total_ms` 的 P50 / P95 | E2E |

### 为什么

P@1 看「第一篇对不对」（客服最敏感）；R@3 看「相关政策有没有进上下文」。只报 Hit@k 会掩盖「第一名是错文档」的失败。关键词命中比同模型 Faithfulness 更独立，面试要并列表。

---

## 6. 意图识别

生产路径：DeepSeek 输出 JSON（`intent_v3`）→ `apply_intent_overlay` 纠偏。LLM 挂了走关键词规则，再走同一覆盖层。

### 6.1 消融（100 条）

| 系统 | 覆盖层 | 准确率 |
|---|---|---|
| 关键词规则 `intent_rules_v1` | 无 | 50.0% |
| `intent_v1` | 关 | 88.0% |
| `intent_v2` | 关 | 87.0% |
| 仅歧义覆盖层 | — | **100%** |
| `intent_v3` + 覆盖（生产） | 开 | **100%** |

规则基线在 `chitchat` / `product_consult` 为 0，`refund` 仅 16.7%：关键词分不清「退货」和「退款到账」。LLM 把整体拉到 87%–88%；覆盖层专门打转人工 vs 退款、政策问法 vs 退换货等稳定中文歧义。

### 6.2 留出集（43 条，未参与调参）

| 系统 | 准确率 |
|---|---|
| 关键词规则 | 39.5% |
| 仅歧义覆盖层 | 39.5%（25 条未覆盖） |
| `intent_v3` + 覆盖 | **95.3%（41/43）** |

两条错误：「拆封的耳机能退吗」被判成 `return_exchange`（金标 `faq_policy`，通用政策 vs 具体售后的边界本身有歧义）；「头疼吃什么药」被判成 `chitchat`（应为 `out_of_scope`）。训练集 100% → 留出集 95.3%，对外引用以留出集为准。

### 代码摘要

```python
# app/agents/nodes/intent_overlay.py
def high_precision_intent(query: str) -> IntentLabel | None:
    # 只在高置信短语上返回标签，否则 None（交给 LLM）
    if any(token in text for token in _HANDOFF):
        return IntentLabel.HUMAN_HANDOFF
    ...
```

评测可 `apply_overlay=False` 做消融；生产 `apply_overlay=True`。

### 为什么 / 面试主动说

训练集 100% **含覆盖层**；43 条没参与调参的留出集是 **95.3%**，对外引用以留出集为准。覆盖针对稳定客服短语（「转人工」压过退款），**不是**按 `i070` 这种 query id 写死金标。规则基线必须保留：它证明「只靠 if-else」不够，也是 LLM 宕机时的兜底。

---

## 7. 检索

65 条，`top_k=5`。稠密侧是本地 `BAAI/bge-small-zh-v1.5`（`BgeM3EmbeddingProvider`，`backend=bge-small-zh-v1.5`）。Cross-Encoder 已加载，报告含 `hybrid+cross-encoder`。8 月更早一轮 hash 对照（报告标识 `all_20260822T042748Z.json`）仅向量 Hit@1 3.1%，该数字仍作为历史对照保留。

### 7.1 系统对照

| 系统 | Precision@1 | Recall@3 | Hit@1 | Hit@3 | MRR |
|---|---|---|---|---|---|
| BM25 | 80.0% | 90.0% | 80.0% | 90.8% | 0.86 |
| 仅 `bge-small-zh-v1.5` | 81.5% | 94.6% | 81.5% | 95.4% | 0.88 |
| BM25 + small + RRF | 83.1% | 94.6% | 83.1% | 95.4% | 0.89 |
| Hybrid + Cross-Encoder | **89.2%** | **96.2%** | 89.2% | 96.9% | **0.93** |

报告字段：取自 `retrieval_20261119T074652Z.json`（未随仓库分发），字段名与 [`eval/runner.py`](../eval/runner.py) 一致。

### 要点

仅 small 相对 BM25 为 P@1 +1.5、R@3 +4.6 个百分点；本轮 RRF 再小幅提高 P@1（+1.6），但不要把 Hybrid 吹成「融合一定大涨」。有效再涨来自 **Cross-Encoder**（相对 BM25：P@1 +9.2，R@3 +6.2；相对 RRF：P@1 +6.1，R@3 +1.6）。与 8 月数字的差异主要来自知识库文档更新，不是模型变化；引用时注明日期与报告标识。

### 7.2 多跳切片（20 条）

| 系统 | P@1 | R@3 | R@5 | Hit@3 | MRR |
|---|---|---|---|---|---|
| BM25 | 90.0% | 75.8% | 83.3% | 100% | 0.95 |
| 仅 `bge-small-zh-v1.5` | 95.0% | 68.3% | 75.8% | 100% | 0.975 |
| BM25 + small + RRF | 85.0% | 78.3% | 83.3% | 100% | 0.925 |
| Hybrid + Cross-Encoder | 90.0% | 76.7% | 80.8% | 100% | 0.95 |

所有系统 **Hit@3 = 100%**（每个问题至少命中一篇相关文档），但多文档覆盖率 **R@5 只有 75.8%–83.3%**：Cross-Encoder 优化的是 top-1 排序，不提升多文档覆盖；RRF 的 P@1 反而降到 85%。多跳的收益主要在生成：RAG 关键词 **95%**，无检索仅 25%。

### 为什么

双塔解决近义召回（「不想要了」≈ 七天无理由）；Cross-Encoder 把 query 和文档拼在一起打分，处理「定制过了七天有质量问题」这类排序。简历写 CE 的 **89.2 / 96.2 / 0.93**，同时说稠密路是 small 不是 m3，且 RRF 增益有限、主要增益来自 Cross-Encoder。

---

## 8. 生成与幻觉

同一 65 条：无检索生成 vs Hybrid 证据生成 vs 忠实度失败后 `self_correct_answer`（先 LLM 重写，再抽取式回退）。生成器是 `cn:deepseek-v4.1-flash`，裁判跑两套：同模型 `deepseek-faithfulness_v1`，以及独立模型 `cn:glm-5.3`（`EVAL_JUDGE_MODEL`）。

同模型裁判：

| 条件 | Faithfulness | 幻觉率 | 关键词命中 |
|---|---|---|---|
| 无检索 | 32.3% | 67.7% | 36.9% |
| RAG | 100% | 0% | 95.4% |
| RAG + LLM 自纠错 | **100%** | **0%** | 95.4% |

独立裁判（`cn:glm-5.3`，判同一批答案）：

| 条件 | Faithfulness | 幻觉率 |
|---|---|---|
| 无检索 | 40.0% | 60.0% |
| RAG | 84.6% | 15.4% |
| RAG + LLM 自纠错 | **89.2%** | **10.8%** |

两裁判在 RAG 答案上的支持判定一致率 **84.6%**。同模型裁判把 RAG 从独立裁判的 84.6% 抬到 100%，说明自评偏乐观约 15 个百分点；自纠错在独立裁判下把支持率从 84.6% 提到 89.2%（+4.6pp），**不是 0 幻觉**。无检索 Faithfulness 仍随裁判模型变化，不能跨模型比较绝对值。

### 代码摘要

```python
# eval/runner.py · eval_generation
naive = await _generate_answer(llm, query, "(无检索结果)")
rag = await _generate_answer(llm, query, evidence)
if not rag_judge.supported:
    checked, checked_judge, _ = await self_correct_answer(...)
```

幻觉定义：`FaithfulnessChecker.check(...).supported is False`。

### 为什么 / 口径陷阱

同模型裁判**偏乐观**（生成器和裁判共享知识与措辞）；独立裁判换了一个模型家族，RAG 支持率从 100% 回落到 **84.6%**，自纠错后 89.2%。但它仍不是人工标注，两裁判一致率只有 84.6%。关键词命中在 RAG 后从 36.9% 到 95.4%，自纠错**没有**再提高关键词——它修的是「证据支撑」，不是补全金标词。对外报数字时同模型/独立裁判两列一起给。

---

## 9. 端到端

16 条走完整 `ChatService` → LangGraph（含护栏、意图、Adaptive 五路、工具、HITL）。限流在评测里临时调到 1000 次/分钟。

| 指标 | 结果 |
|---|---|
| 拒答准确率（3 条） | **100%** |
| 关键词命中（11 条） | **11/11（100%）** |
| 工具路径（5 条） | **100%** |
| 延迟 P50 / P95 / mean | **8083 / 9213 / 6864 ms** |
| 意图分段 P50（n=15） | 1559 ms |
| 生成 LLM 分段 P50（n=13） | 1921 ms |
| 检索分段 P50（n=6） | 1640 ms |
| 工具分段 P50（n=4） | 4.7 ms |

分阶段 P50 只统计实际执行该阶段的 case，并同时输出样本数（`retrieval_n=6`、`tool_n=4`），未检索的 case 不再把中位数拉到 0。n=16 的 P95 是离散分位（9.2s）；实际最慢是 `e02` multi **12.7s**。注入 `e10` **14.6 ms**——正则拦下，不打 LLM。

### 策略分布（实测）

| ID | 意图 | 策略 | 延迟 |
|---|---|---|---|
| e01 | greeting | direct | 4.2s |
| e02 | return_exchange | multi | 12.7s |
| e03 | faq_policy | single | 9.1s |
| e04 | faq_policy | multi | 9.2s |
| e05 | faq_policy | single | 8.6s |
| e06 | order_inquiry | agent | 7.8s |
| e07 | order_inquiry | agent | 8.1s |
| e08 | logistics | agent | 9.0s |
| e09 | coupon | single | 7.7s |
| e10 | （注入） | refuse | 15ms |
| e11 | out_of_scope | refuse | 3.2s |
| e12 | out_of_scope | refuse | 2.9s |
| e13 | human_handoff | direct | 4.7s |
| e14 | product_consult | agent | 8.3s |
| e15 | cancel_order | agent | 5.3s（HITL 确认，未真取消） |
| e16 | faq_policy | single | 9.0s |

来源：`e2e_20261119T075759Z.json`（未随仓库分发）；汇总口径见 [`MEASURED.md`](../eval/MEASURED.md)。

---

## 10. 自动化测试

`pytest -q`：**202 passed**（本地含 `.[rerank]`；CI 干净环境 **198 passed + 4 skipped**，跳过的是需要可选依赖的模型加载测试）。覆盖 RRF、Adaptive 路由、护栏、压缩、工具权限、Skill、MCP、执行 Fallback、自纠错、订单仓储、HTTP 聊天、会话归属、待确认动作持久化、知识库热重载、启动迁移、配额解析、可观测性资产，以及 `eval` 指标函数与检索缓存 / 自纠正 / 查询改写 / 向量检索 / 分词。

pytest **不**调用真实 DeepSeek / 不下载 Cross-Encoder。线上指标必须 `eval.runner --llm`；CE 分数必须 `hybrid+cross-encoder` 那一行。

---

## 11. 与简历数字对照

**可以写（已测量）：**

- 意图：规则 50% → 无覆盖 v1 88% / v2 87% → v3+覆盖训练集 **100%**、留出集 **95.3%**
- 仅 `bge-small`：P@1 **81.5%**、R@3 **94.6%**、MRR **0.88**
- Hybrid+CE：P@1 **89.2%**、R@3 **96.2%**、MRR **0.93**
- BM25 对照：P@1 **80.0%**（58–85）、R@3 **90%**（63–91）、MRR **0.86**
- 幻觉：同模型 无检索 67.7% → RAG+自纠错 **0%**；独立裁判 60% → **10.8%**；两裁判 RAG 一致率 **84.6%**
- 无检索 Faithfulness **32.3%**（同模型）/ **40.0%**（独立裁判）
- 多跳 20 条：Hit@3 **100%**、Hybrid+CE R@5 **80.8%**；RAG 关键词 **95%**
- E2E P50 **8.08s** / P95 **9.21s**；注入拒答约 15ms；关键词 **11/11**

**不能写：**

- 把无检索 Faithfulness 跨裁判/跨模型直接比较
- 把训练集 100% 当成泛化 100%（留出集 95.3%）
- 把 100% 意图说成「纯 LLM、无规则」
- 把 Faithfulness 100% 说成「人工事实核查零幻觉」
- 把稠密路写成 bge-m3，或把 RRF 说成本集上显著优于仅 small
- 把 E2E 的 8.08s 说成线上 SLA

---

## 12. 未测项与下一步

| 项 | 状态 | 怎样才算测完 |
|---|---|---|
| Cross-Encoder `BAAI/bge-reranker-base` | **已测** Hybrid+CE P@1 89.2% / R@3 96.2% | 换 checkpoint 后重跑 retrieval |
| `bge-small-zh-v1.5` | **已测** 仅向量 P@1 81.5% / R@3 94.6% | 换更大 embedding 后另跑，勿混写成 m3 |
| 意图无覆盖的 v3 单列 | 训练集 100% / 留出集 95.3% 都含覆盖；关覆盖的 v3 未单独落 JSON | `apply_overlay=False` 再跑一列 |
| 意图留出集 | **已测** 43 条 95.3% | 扩到 100+ 条，并复核 2 条歧义金标 |
| 多跳多文档覆盖 | **已测** R@5 80.8%，仍不完整 | 提高 rerank_top_n 或按问题拆子查询后重跑 |
| E2E 关键词 | 本轮 **11/11** | 换模型或改路由后重跑 |
| 人工 Faithfulness | 仍缺；现有同模型 + 独立模型两套裁判，RAG 一致率 84.6% | 抽 20 条双人标注 |
| 并发 / 限流 / 故障注入 | 仅单请求 E2E | 另做压测，不进本报告 |

---

上一章 [13 · 面试稿](13-interview-engineering-rag.md) · [返回目录](README.md)
