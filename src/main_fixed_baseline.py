# src/main_agentcoder.py
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset

from src.agents.llm_client import AsyncLLMClient
from src.agents.programmer import ProgrammerAgent
from src.agents.test_designer import TestDesignerAgent
from src.execution.test_executor import TestExecutor


# ============================================================================
# CONFIG
# ============================================================================
BASELINE_FILENAME = "baseline.jsonl"
REFINE_ROUNDS_LIST = [1 , 2 , 3]


def find_data_file(filename: str) -> Path:
    """
    Find a file in common data directories.
    Compatible with running from project root or src directory.
    """
    current_file = Path(__file__).resolve()

    candidate_paths = [
        Path.cwd() / "data" / filename,
        current_file.parents[1] / "data" / filename,
        current_file.parent / "data" / filename,
    ]

    for path in candidate_paths:
        if path.exists():
            return path

    searched = "\n".join(str(p) for p in candidate_paths)
    raise FileNotFoundError(f"Cannot find {filename}. Searched:\n{searched}")


def load_baseline_completions(filepath: Path) -> Dict[str, str]:
    """
    Load baseline completions from a jsonl file.
    Expected format:
    {"task_id": "...", "completion": "..."}
    """
    completions = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            task_id = item["task_id"]
            completion = item.get("completion", "")
            completions[task_id] = completion

    return completions


def build_result(
    task_id: str,
    completion: str,
    test_cases: str,
    passed_designer: bool,
    attempts_used: int,
    refine_rounds: int,
    status: str,
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "completion": completion,
        "generated_tests": test_cases,
        "passed_designer_tests": passed_designer,
        "attempts_used": attempts_used,
        "refine_rounds": refine_rounds,
        "status": status,
    }


# ============================================================================
# CORE PIPELINE
# ============================================================================
async def process_single_task(
    task: Dict[str, Any],
    baseline_completions: Dict[str, str],
    programmer: ProgrammerAgent,
    designer: TestDesignerAgent,
    executor: TestExecutor,
    semaphore: asyncio.Semaphore,
    refine_rounds_list: List[int],
) -> Dict[int, Dict[str, Any]]:
    """
    Run refinement from a fixed baseline completion.

    Returns:
        {
            1: result_after_refine1,
            2: result_after_refine2,
            3: result_after_refine3,
        }
    """
    async with semaphore:
        task_id = task["task_id"]
        prompt = task["prompt"]
        max_reflections = max(refine_rounds_list)

        print(f"[Task Start] Processing {task_id} from baseline...")

        # Step 1: Use baseline completion as initial code
        current_code = baseline_completions.get(task_id, "")

        if not current_code:
            print(f"[Warning] Baseline completion is empty or missing for {task_id}.")
            empty_results = {}

            for r in refine_rounds_list:
                empty_results[r] = build_result(
                    task_id=task_id,
                    completion="",
                    test_cases="",
                    passed_designer=False,
                    attempts_used=0,
                    refine_rounds=r,
                    status="empty_or_missing_baseline",
                )

            return empty_results

        # Step 2: Generate test cases once
        test_cases_response = await designer.generate_tests(task_id, prompt)
        test_cases = test_cases_response.get("tests", "")

        results_by_round = {}

        # Step 3: Execute and refine
        for attempt in range(1, max_reflections + 1):
            exec_result = await asyncio.to_thread(
                executor.run_single_test, task_id, current_code, test_cases
            )

            if exec_result.get("passed"):
                print(
                    f"[Task Success] {task_id} passed designer tests before refine round {attempt}."
                )

                # If already passed, later refine-k results remain the same code
                for r in refine_rounds_list:
                    if r >= attempt and r not in results_by_round:
                        results_by_round[r] = build_result(
                            task_id=task_id,
                            completion=current_code,
                            test_cases=test_cases,
                            passed_designer=True,
                            attempts_used=attempt - 1,
                            refine_rounds=r,
                            status="passed_designer_tests",
                        )

                break

            print(
                f"[Refining] {task_id} failed designer tests. "
                f"Refining code (Attempt {attempt}/{max_reflections})..."
            )

            refined_code = await programmer.refine_code(
                problem_description=prompt,
                original_code=current_code,
                error_feedback=exec_result["feedback"],
            )

            if refined_code:
                current_code = refined_code

                if attempt in refine_rounds_list:
                    results_by_round[attempt] = build_result(
                        task_id=task_id,
                        completion=current_code,
                        test_cases=test_cases,
                        passed_designer=False,
                        attempts_used=attempt,
                        refine_rounds=attempt,
                        status="refined",
                    )
            else:
                print(
                    f"[Warning] Refinement returned empty for {task_id}. "
                    f"Keeping previous code for remaining rounds."
                )

                for r in refine_rounds_list:
                    if r >= attempt and r not in results_by_round:
                        results_by_round[r] = build_result(
                            task_id=task_id,
                            completion=current_code,
                            test_cases=test_cases,
                            passed_designer=False,
                            attempts_used=attempt,
                            refine_rounds=r,
                            status="refinement_empty_keep_previous",
                        )

                break

        # Fill any missing round results with the latest code
        for r in refine_rounds_list:
            if r not in results_by_round:
                results_by_round[r] = build_result(
                    task_id=task_id,
                    completion=current_code,
                    test_cases=test_cases,
                    passed_designer=False,
                    attempts_used=max_reflections,
                    refine_rounds=r,
                    status="max_reflections_reached",
                )

        return results_by_round


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================
async def main():
    print("--- Starting AgentCoder Refinement from Baseline ---")

    # 1. Initialize Clients and Agents
    llm_client = AsyncLLMClient()
    programmer = ProgrammerAgent(llm_client)
    designer = TestDesignerAgent(llm_client=llm_client)
    executor = TestExecutor()

    # 2. Setup Concurrency Control
    max_concurrent_tasks = 5
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    # 3. Load baseline completions
    baseline_path = find_data_file(BASELINE_FILENAME)
    print(f"Loading baseline completions from: {baseline_path}")
    baseline_completions = load_baseline_completions(baseline_path)

    # 4. Load HumanEval Dataset
    print("Loading HumanEval dataset...")
    try:
        dataset = load_dataset("openai_humaneval", split="test")
        tasks = [
            entry for entry in dataset
            if entry["task_id"] in baseline_completions
        ]
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        return

    print(f"Total baseline problems to process: {len(tasks)}")
    print(f"Refine rounds to output: {REFINE_ROUNDS_LIST}")

    # 5. Build and Run Async Tasks
    coroutines = [
        process_single_task(
            task=task,
            baseline_completions=baseline_completions,
            programmer=programmer,
            designer=designer,
            executor=executor,
            semaphore=semaphore,
            refine_rounds_list=REFINE_ROUNDS_LIST,
        )
        for task in tasks
    ]

    print("Executing refinement pipeline...")
    task_results = await asyncio.gather(*coroutines)

    # 6. Group results by refine round
    results_by_round = {r: [] for r in REFINE_ROUNDS_LIST}

    for result_dict in task_results:
        for r in REFINE_ROUNDS_LIST:
            results_by_round[r].append(result_dict[r])

    # 7. Save jsonl files
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    for r in REFINE_ROUNDS_LIST:
        output_file = os.path.join(output_dir, f"agentcoder_refine{r}.jsonl")

        with open(output_file, "w", encoding="utf-8") as f:
            for res in results_by_round[r]:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

        print(f"[Saved] Refine{r} results saved to {output_file}")

    print("\n--- Refinement Pipeline Complete! ---")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Interrupted] Evaluation stopped by user.")