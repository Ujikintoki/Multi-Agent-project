# eval.py
import json
import subprocess
import tempfile
import os
import sys
from datasets import load_dataset


def load_results(jsonl_path):
    results = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            results[item["task_id"]] = item["completion"]
    return results


def run_official_test(task_id, completion, official_test, entry_point):
    """使用官方测试用例在沙盒中运行最终评估"""
    # 拼接格式：生成的代码 + 官方测试代码 + 触发函数调用的 assert
    full_code = f"{completion}\n\n{official_test}\n\ncheck({entry_point})"

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py")
    os.close(tmp_fd)

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(full_code)

        process = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=5,  # 官方测试通常很快，超时设为5秒防死循环
        )
        return process.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        return False
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    print("Loading HumanEval dataset for official tests...")
    dataset = load_dataset("openai_humaneval", split="test")
    official_data = {item["task_id"]: item for item in dataset}

    results_path = "data/agentcoder_results.jsonl"
    print(f"Loading generated completions from {results_path}...")
    completions = load_results(results_path)

    passed_count = 0
    total_count = len(completions)

    if total_count == 0:
        print("No results found. Please run main.py first.")
        return

    print("Running True Pass@1 Evaluation...")
    for task_id, completion in completions.items():
        if not completion:
            continue

        official_task = official_data[task_id]
        official_test = official_task["test"]
        entry_point = official_task["entry_point"]

        is_passed = run_official_test(task_id, completion, official_test, entry_point)
        if is_passed:
            passed_count += 1

    pass_at_1 = (passed_count / total_count) * 100
    print(f"\n{'=' * 30}")
    print(f"Total Evaluated: {total_count}")
    print(f"Total Passed:    {passed_count}")
    print(f"True Pass@1:     {pass_at_1:.2f}%")
    print(f"{'=' * 30}")


if __name__ == "__main__":
    main()
