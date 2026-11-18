import { describe, expect, it } from "vitest";
import type { Capabilities } from "../types";
import { runtimeStatus } from "./runtime";

const base: Capabilities = {
  database: "sqlite",
  cache: "memory",
  vector_store: "memory",
  retriever: "hybrid",
  reranker: "cross-encoder",
  llm: "deepseek",
  llm_model: "cn:deepseek-v4.1-flash",
  embedding_model: "bge-small-zh-v1.5",
  reranker_model: "BAAI/bge-reranker-base",
  llm_quota_used_percent: null,
  llm_quota_reset_at: null,
  degraded: false,
  degraded_reasons: [],
};

describe("runtimeStatus", () => {
  it("shows the active model and backend details in the tooltip", () => {
    const status = runtimeStatus(base);
    expect(status?.label).toBe("deepseek-v4.1-flash");
    expect(status?.degraded).toBe(false);
    expect(status?.title).toContain("bge-small-zh-v1.5");
    expect(status?.title).toContain("BAAI/bge-reranker-base");
    expect(status?.title).toContain("存储：memory");
    expect(status?.title).toContain("缓存：memory");
  });

  it("marks offline mode and lists degradation reasons", () => {
    const status = runtimeStatus({
      ...base,
      llm: "offline",
      llm_model: "offline",
      reranker: "lexical",
      reranker_model: "lexical",
      degraded: true,
      degraded_reasons: ["llm: offline (no api key)", "reranker: lexical fallback"],
    });
    expect(status?.label).toBe("离线模式");
    expect(status?.degraded).toBe(true);
    expect(status?.title).toContain("llm: offline");
  });

  it("shows the remaining quota when the provider reports one", () => {
    const status = runtimeStatus({
      ...base,
      llm_quota_used_percent: 23,
      llm_quota_reset_at: 1795071817,
    });
    expect(status?.label).toContain("77%");
    expect(status?.title).toContain("已用 23%");
    expect(status?.title).toContain("剩余 77%");
  });

  it("returns null before capabilities load", () => {
    expect(runtimeStatus(null)).toBeNull();
  });
});
