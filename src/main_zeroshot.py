# src/main_zeroshot.py
import asyncio
import json
import os
from typing import Any, Dict

from datasets import load_dataset

from src.agents.llm_client import AsyncLLMClient
from src.agents.programmer import ProgrammerAgent
# 注意：我们这里甚至都不需要导入 Designer 和 Executor 了


# ============================================================================
# CORE PIPELINE (Zero-shot Baseline)
# ============================================================================
async def process_single_task(
    task: Dict[str, Any],
    programmer: ProgrammerAgent,
    semaphore: asyncio.Semaphore,
) -> Dict[str, str]:
    """
    Executes ONLY the Programmer Agent for a single HumanEval problem (Zero-shot).
    Wrapped in a semaphore to strictly control concurrent API requests.
    """
    async with semaphore:
        task_id = task["task_id"]
        prompt = task["prompt"]

        print(f"[Task Start] Processing {task_id} (Zero-shot)...")

        # Step 1: Initial Code Generation ONLY
        current_code = await programmer.generate_initial_code(prompt)

        # Fallback if generation failed entirely
        if not current_code:
            print(f"[Warning] Failed to generate initial code for {task_id}.")
            return {
                "task_id": task_id,
                "completion": "",
                "generated_tests": "",
                "passed_designer_tests": False,
                "attempts_used": 1,
            }

        print(f"[Task Success] {task_id} generated code.")

        # 直接打包返回，填充默认的空数据以兼容 eval_advanced.py
        return {
            "task_id": task_id,
            "completion": current_code,
            "generated_tests": "N/A",  # 没跑 Designer，填 N/A
            "passed_designer_tests": False,  # 没跑本地测试，默认 False
            "attempts_used": 1,  # Zero-shot 只用了一次尝试
        }


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================
async def main():
    print("--- Starting Zero-shot Baseline Evaluation ---")

    # 1. Initialize Clients and Agents (只初始化 Programmer)
    llm_client = AsyncLLMClient()
    programmer = ProgrammerAgent(llm_client)

    # 2. Setup Concurrency Control
    max_concurrent_tasks = 10
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    # 3. Load HumanEval Dataset
    print("Loading HumanEval dataset...")
    try:
        dataset = load_dataset("openai_humaneval", split="test")
        tasks = [entry for entry in dataset]
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        return

    print(f"Total problems to process: {len(tasks)}")

    # 4. Build and Run Async Tasks (去掉了 designer 和 executor 参数)
    coroutines = [process_single_task(task, programmer, semaphore) for task in tasks]

    print("Executing Zero-shot pipeline...")
    results = await asyncio.gather(*coroutines)

    # 5. Save Results for Pass@1 Evaluation
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    # 【核心改动】修改了输出文件名，以区分与 main.py 的结果
    output_file = os.path.join(output_dir, "zeroshot_baseline_results.jsonl")

    with open(output_file, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

    print(f"\n--- Baseline Complete! Results saved to {output_file} ---")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Interrupted] Evaluation stopped by user.")
