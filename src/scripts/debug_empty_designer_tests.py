import asyncio
from datasets import load_dataset

from agents.test_designer import TestDesignerAgent

EMPTY_IDS = [
    "HumanEval/32",
    "HumanEval/81",
    "HumanEval/93",
    "HumanEval/145",
]


async def main():
    dataset = load_dataset("openai_humaneval", split="test")
    problem_map = {sample["task_id"]: sample for sample in dataset}

    agent = TestDesignerAgent(debug=True)

    for task_id in EMPTY_IDS:
        print("\n" + "=" * 80)
        print(f"DEBUGGING {task_id}")
        print("=" * 80)

        sample = problem_map[task_id]
        result = await agent.generate_tests(
            task_id=task_id,
            problem_prompt=sample["prompt"],
        )

        print("\nFINAL CLEANED TESTS:")
        print(result["tests"])
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())