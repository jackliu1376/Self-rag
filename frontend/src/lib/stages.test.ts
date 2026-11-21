import { describe, expect, it } from "vitest";
import { stageLabel } from "./stages";

describe("stageLabel", () => {
  it("maps graph nodes to user-facing stages", () => {
    expect(stageLabel("retrieve_docs")).toBe("检索知识");
    expect(stageLabel("generate_answer")).toBe("生成回答");
    expect(stageLabel("check_faithfulness")).toBe("校验答案");
  });

  it("falls back for unknown nodes", () => {
    expect(stageLabel("future_node")).toBe("处理中");
  });
});
