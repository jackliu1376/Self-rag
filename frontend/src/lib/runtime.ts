import type { Capabilities } from "../types";

export interface RuntimeStatus {
  label: string;
  title: string;
  degraded: boolean;
}

export function runtimeStatus(capabilities: Capabilities | null): RuntimeStatus | null {
  if (!capabilities) return null;
  const model = capabilities.llm_model === "offline"
    ? "离线模式"
    : capabilities.llm_model.split(":").pop() || capabilities.llm_model;
  const used = capabilities.llm_quota_used_percent;
  const remaining = used === null ? null : Math.max(0, Math.min(100, 100 - used));
  const reset = capabilities.llm_quota_reset_at
    ? new Date(capabilities.llm_quota_reset_at * 1000).toLocaleString("zh-CN", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    })
    : null;
  const label = remaining === null ? model : `${model} · ${remaining}%`;
  const title = [
    `模型：${capabilities.llm_model}`,
    `向量：${capabilities.embedding_model}`,
    `精排：${capabilities.reranker_model}`,
    `检索：${capabilities.retriever}`,
    `存储：${capabilities.vector_store}`,
    `缓存：${capabilities.cache}`,
    ...(remaining === null
      ? []
      : [`额度（最近一次调用）：已用 ${used}%（剩余 ${remaining}%）${reset ? `，${reset} 重置` : ""}`]),
    ...capabilities.degraded_reasons.map((reason) => `降级：${reason}`),
  ].join("\n");
  return { label, title, degraded: capabilities.degraded };
}
