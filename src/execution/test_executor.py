import json
import os
import subprocess
import tempfile
import logging
import sys
import uuid  # 用于生成唯一的临时文件名
from typing import Dict, List, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestExecutor:
    """
    Test Executor Agent: 负责执行代码并返回反馈。
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def load_jsonl(self, filepath: str) -> Dict[str, str]:
        """加载 jsonl 文件并返回以 task_id 为键的字典"""
        data = {}
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return data

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                # 处理不同字段名：Programmer 输出通常是 'completion'，Designer 是 'tests'
                content = item.get("completion") or item.get("tests") or ""
                data[item["task_id"]] = content
        return data

    def run_single_test(self, task_id: str, code: str, tests: str) -> Dict:
        """
        运行单个任务的测试，返回包含结果和反馈的字典。
        """
        # 1. 判断测试用例是否为空
        if not tests.strip():
            logger.info(f"Test cases for task {task_id} are empty, skipping execution.")
            return {
                "task_id": task_id,
                "passed": False,
                "feedback": "No test cases provided by Test Designer.",
                "status": "skipped",
            }

        # 2. 拼接完整代码
        full_code = f"{code}\n\n{tests}"

        # 3. 在 Windows 上更稳健的临时文件处理方式
        # 手动在系统临时目录创建一个文件，确保在运行 subprocess 前关闭它
        temp_dir = tempfile.gettempdir()
        filename = f"agent_test_{uuid.uuid4().hex}.py"
        tmp_path = os.path.join(temp_dir, filename)

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(full_code)

            # 确保文件已写入并关闭，现在启动子进程
            result_info = {
                "task_id": task_id,
                "passed": False,
                "feedback": "",
                "status": "failed",
            }

            process = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if process.returncode == 0:
                result_info["passed"] = True
                result_info["status"] = "success"
                result_info["feedback"] = "All tests passed successfully."
            else:
                # 捕获详细的错误信息
                error_msg = process.stderr if process.stderr else process.stdout
                if not error_msg:
                    error_msg = f"Process exited with code {process.returncode} but no error message."
                result_info["feedback"] = f"Test failed with error:\n{error_msg}"

        except subprocess.TimeoutExpired:
            result_info["feedback"] = (
                f"Execution timed out after {self.timeout} seconds."
            )
        except Exception as e:
            result_info["feedback"] = f"Executor internal error: {str(e)}"
        finally:
            # 运行结束后清理临时文件
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass

        return result_info


#     def execute_all(self, code_path: str, tests_path: str, output_path: str):
#         """
#         主执行流程：读取代码和测试，运行并保存反馈结果。
#         """
#         logger.info(f"Starting test execution process...")

#         programmer_codes = self.load_jsonl(code_path)
#         designer_tests = self.load_jsonl(tests_path)

#         results = []
#         task_ids = set(programmer_codes.keys()).union(set(designer_tests.keys()))

#         for task_id in sorted(task_ids):
#             code = programmer_codes.get(task_id, "")
#             tests = designer_tests.get(task_id, "")

#             if not code:
#                 logger.warning(f"Task {task_id} is missing code completion, skipping.")
#                 continue

#             logger.info(f"Executing task: {task_id}")
#             res = self.run_single_test(task_id, code, tests)
#             results.append(res)

#         # 确保输出目录存在
#         output_dir = os.path.dirname(output_path)
#         if output_dir and not os.path.exists(output_dir):
#             os.makedirs(output_dir)

#         with open(output_path, 'w', encoding='utf-8') as f:
#             for r in results:
#                 f.write(json.dumps(r) + "\n")

#         success_count = sum(1 for r in results if r["passed"])
#         logger.info(f"Execution complete. Success: {success_count}/{len(results)}")
#         return results

# if __name__ == "__main__":
#     # 使用绝对路径确保在不同环境下运行都能找到数据
#     base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     executor = TestExecutor(timeout=10)

#     executor.execute_all(
#         code_path=os.path.join(base_dir, "data", "baseline_samples.jsonl"),
#         tests_path=os.path.join(base_dir, "data", "designer_tests.jsonl"),
#         output_path=os.path.join(base_dir, "data", "executor_feedback.jsonl")
#     )
