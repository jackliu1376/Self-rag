const STAGE_LABELS: Record<string, string> = {
  apply_guardrail: "安全检查",
  classify_intent: "识别意图",
  route_complexity: "选择策略",
  retrieve_docs: "检索知识",
  plan_tasks: "规划工具",
  execute_tools: "执行工具",
  generate_answer: "生成回答",
  refuse_answer: "拒答",
  check_faithfulness: "校验答案",
  compact_memory: "整理记忆",
};

export function stageLabel(node: string): string {
  return STAGE_LABELS[node] ?? "处理中";
}
