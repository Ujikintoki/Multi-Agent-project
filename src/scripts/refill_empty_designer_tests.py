import os
import json
import asyncio
from datasets import load_dataset
from tqdm.asyncio import tqdm

from agents.test_designer import TestDesignerAgent


INPUT_PATH = "../data/designer_tests.jsonl"
OUTPUT_PATH = "../data/designer_tests.jsonl"


async def main():
    # 1. 读取现有 designer_tests.jsonl
    existing_results = []
    empty_ids = []

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            existing_results.append(item)
            if not item.get("tests", "").strip():
                empty_ids.append(item["task_id"])

    print(f"Found {len(empty_ids)} empty test entries.")
    if not empty_ids:
        print("No empty entries found. Nothing to refill.")
        return

    # 2. 加载 HumanEval 原题
    print("Loading HumanEval dataset...")
    dataset = load_dataset("openai_humaneval", split="test")
    problem_map = {sample["task_id"]: sample for sample in dataset}

    # 3. 初始化 agent
    agent = TestDesignerAgent(debug=False)

    # 4. 只补跑空题
    print("Refilling empty test entries...")
    refill_map = {}

    for task_id in tqdm(empty_ids, desc="Refilling empty designer tests"):
        sample = problem_map[task_id]
        result = await agent.generate_tests(
            task_id=task_id,
            problem_prompt=sample["prompt"],
        )
        refill_map[task_id] = result["tests"]

    # 5. 覆盖原结果
    updated_results = []
    refill_success = 0

    for item in existing_results:
        task_id = item["task_id"]
        if task_id in refill_map and refill_map[task_id].strip():
            item["tests"] = refill_map[task_id]
            refill_success += 1
        updated_results.append(item)

    # 6. 写回文件
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in updated_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Refill complete. Successfully updated {refill_success} entries.")
    print(f"Saved updated file to: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())