# src/main_agentcoder.py
import asyncio
import json
import os
from typing import Any, Dict

from datasets import load_dataset

from src.agents.llm_client import AsyncLLMClient
from src.agents.programmer import ProgrammerAgent

from src.agents.test_designer import TestDesignerAgent
from src.test_executor import TestExecutor


# ============================================================================
# MOCK CLASSES (To be replaced when you implement the actual modules)
# ============================================================================
# class MockTestDesigner:
#     async def generate_tests(self, problem_prompt: str) -> str:
#         """Placeholder for the Test Designer Agent."""
#         return "assert True # Mock test case"


# class MockExecutor:
#     def execute(self, code: str, tests: str) -> Dict[str, Any]:
#         """Placeholder for the Python execution sandbox."""
#         # Assume it passes for the sake of pipeline testing
#         return {"success": True, "feedback": "All tests passed."}
designer = TestDesignerAgent(debug=False)
executor = TestExecutor(timeout=10)


# ============================================================================
# CORE PIPELINE
# ============================================================================
async def process_single_task(
    task: Dict[str, Any],
    programmer: ProgrammerAgent,
    designer: TestDesignerAgent,
    executor: TestExecutor,
    semaphore: asyncio.Semaphore,
    # max_reflections: int = 3,
    max_reflections: int = 1,
) -> Dict[str, str]:
    """
    Executes the full AgentCoder pipeline for a single HumanEval problem.
    Wrapped in a semaphore to strictly control concurrent API requests.
    """
    async with semaphore:
        task_id = task["task_id"]
        prompt = task["prompt"]

        print(f"[Task Start] Processing {task_id}...")

        # Step 1: Initial Code Generation
        current_code = await programmer.generate_initial_code(prompt)

        # Fallback if generation failed entirely
        if not current_code:
            print(f"[Warning] Failed to generate initial code for {task_id}.")
            return {"task_id": task_id, "completion": ""}

        # Step 2: Test Case Generation
        # test_cases = await designer.generate_tests(prompt)
        test_cases_response = await designer.generate_tests(task_id, prompt)
        test_cases = test_cases_response.get("tests", "")

        # Step 3: Execution and Reflection Loop
        for attempt in range(max_reflections):
            # exec_result = executor.execute(current_code, test_cases)
            exec_result = await asyncio.to_thread(
                executor.run_single_test, task_id, current_code, test_cases
            )
            # if exec_result["success"]:
            #     print(
            #         f"[Task Success] {task_id} passed tests on attempt {attempt + 1}."
            #     )
            #     break
            if exec_result.get("passed"):  # 确保这里用的是 passed 而不是 success
                print(
                    f"[Task Success] {task_id} passed tests on attempt {attempt + 1}."
                )
                break

            print(
                f"[Refining] {task_id} failed. Refining code (Attempt {attempt + 1}/{max_reflections})..."
            )

            # Request refined code based on execution feedback
            refined_code = await programmer.refine_code(
                problem_description=prompt,
                original_code=current_code,
                error_feedback=exec_result["feedback"],
            )

            if refined_code:
                current_code = refined_code
            else:
                print(
                    f"[Warning] Refinement returned empty for {task_id}. Keeping previous code."
                )
                break

        # return {"task_id": task_id, "completion": current_code}
        passed_designer = (
            exec_result.get("passed", False) if "exec_result" in locals() else False
        )

        # 将更丰富的运行数据打包返回
        return {
            "task_id": task_id,
            "completion": current_code,
            "generated_tests": test_cases,  # 用于评估 Metric 4 (测试生成有效率)
            "passed_designer_tests": passed_designer,  # 用于评估 Metric 2 (假阳性率)
            "attempts_used": attempt + 1,  # 用于评估 Metric 3 (平均反思步数)
        }


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================
async def main():
    print("--- Starting AgentCoder Evaluation Pipeline ---")

    # 1. Initialize Clients and Agents
    llm_client = AsyncLLMClient()
    programmer = ProgrammerAgent(llm_client)
    designer = TestDesignerAgent()
    executor = TestExecutor()

    # 2. Setup Concurrency Control
    # Limit to 10 concurrent tasks to avoid hitting Azure OpenAI rate limits
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

    # 4. Build and Run Async Tasks
    # Using asyncio.gather to run tasks concurrently while respecting the semaphore
    coroutines = [
        process_single_task(task, programmer, designer, executor, semaphore)
        for task in tasks
    ]

    print("Executing pipeline...")
    results = await asyncio.gather(*coroutines)

    # 5. Save Results for Pass@1 Evaluation
    # The standard format for HumanEval eval is JSONL (one JSON object per line)
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "agentcoder_results_refine1time.jsonl")

    with open(output_file, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

    print(f"\n--- Pipeline Complete! Results saved to {output_file} ---")


if __name__ == "__main__":
    # Prevent nested event loop issues if running in specific environments
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Interrupted] Evaluation stopped by user.")
