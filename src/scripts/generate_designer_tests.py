import os
import json
import asyncio
from datasets import load_dataset
from tqdm.asyncio import tqdm

from agents.test_designer import TestDesignerAgent


async def main():
    print("Loading HumanEval dataset...")
    dataset = load_dataset("openai_humaneval", split="test")

    agent = TestDesignerAgent(debug=False)
    results = []

    print(f"Loaded {len(dataset)} HumanEval tasks. Generating tests...")

    for sample in tqdm(dataset, desc="Generating designer tests"):
        task_id = sample["task_id"]
        problem_prompt = sample["prompt"]

        result = await agent.generate_tests(
            task_id=task_id,
            problem_prompt=problem_prompt,
        )

        results.append(result)

    output_dir = "../data"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "designer_tests.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} designer test sets to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())