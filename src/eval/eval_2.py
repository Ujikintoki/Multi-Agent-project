import json
import os
import subprocess
import tempfile
from datasets import load_dataset


def run_local_eval():
    print("Step 1: loading HumanEval dataset...")
    dataset = load_dataset("openai_humaneval", split="test")
    print(f"Step 2: loaded {len(dataset)} problems.")

    jsonl_path = "../data/baseline_samples.jsonl"
    completions = {}

    print("Step 3: reading baseline_samples.jsonl...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            completions[data["task_id"]] = data["completion"]
    print(f"Step 4: loaded {len(completions)} completions.")

    passed = 0
    total = 0

    print("Step 5: starting evaluation...")

    for problem in dataset:
        task_id = problem["task_id"]
        prompt = problem["prompt"]
        completion = completions.get(task_id, "")
        test_code = problem["test"]
        entry_point = problem["entry_point"]

        full_code = prompt + completion + "\n" + test_code + f"\ncheck({entry_point})\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp:
            tmp.write(full_code)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                passed += 1
        except subprocess.TimeoutExpired:
            pass
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        total += 1

        if total % 20 == 0:
            print(f"Progress: {total}/{len(dataset)}")

    pass_at_1 = passed / total if total > 0 else 0.0

    print("\n==================================")
    print(f"Passed: {passed}/{total}")
    print(f"Pass@1: {pass_at_1:.4f}")
    print("==================================")


if __name__ == "__main__":
    run_local_eval()
