"""
Evaluation Pipeline for Multi-Agent Code Generation.
Evaluates completions against the HumanEval dataset.

Usage:
    python3 eval.py --input_path data/agentcoder_results.jsonl --output_dir results/
"""

import argparse
import ast
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from datasets import load_dataset


# ============================================================================
# 1. Configuration & Setup
# ============================================================================
def setup_logger(log_level: int = logging.INFO) -> logging.Logger:
    """Configures a standard console logger."""
    logger = logging.getLogger("EvalPipeline")
    logger.setLevel(log_level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = setup_logger()


@dataclass
class EvalMetrics:
    """Dataclass to enforce rigid structure for our evaluation metrics."""

    total_tasks: int = 0
    true_passed: int = 0
    syntax_correct: int = 0
    valid_tests_generated: int = 0
    false_positives: int = 0
    designer_passed_total: int = 0
    total_attempts_for_success: int = 0


# ============================================================================
# 2. Evaluation Engine
# ============================================================================
class HumanEvalEvaluator:
    """
    Evaluator class for assessing code completions against the HumanEval dataset.
    """

    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.metrics = EvalMetrics()
        logger.info("Loading HumanEval dataset from HuggingFace...")
        try:
            dataset = load_dataset("openai_humaneval", split="test")
            self.official_data: Dict[str, Dict[str, Any]] = {
                item["task_id"]: item for item in dataset
            }
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    @staticmethod
    def _check_syntax(code_string: str) -> bool:
        """Statically checks Python syntax without execution."""
        if not code_string:
            return False
        try:
            ast.parse(code_string)
            return True
        except SyntaxError:
            return False

    @staticmethod
    def _check_test_validity(test_string: str) -> bool:
        """Validates if the generated test contains valid assertions."""
        if not test_string or "assert " not in test_string:
            return False
        try:
            ast.parse(test_string)
            return True
        except SyntaxError:
            return False

    def _run_sandbox(
        self, completion: str, official_test: str, entry_point: str
    ) -> bool:
        """
        Executes the completion and official tests in an isolated subprocess.
        """
        full_code = f"{completion}\n\n{official_test}\n\ncheck({entry_point})"

        # Secure temporary file handling
        fd, tmp_path = tempfile.mkstemp(suffix=".py", text=True)
        os.close(fd)

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(full_code)

            process = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return process.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception as e:
            logger.debug(f"Sandbox execution error: {e}")
            return False
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def evaluate_file(self, input_path: Path) -> None:
        """Reads the generated results and calculates all metrics."""
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            raise FileNotFoundError(f"Missing file: {input_path}")

        logger.info(f"Starting evaluation on: {input_path}")

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON line.")
                    continue

                self.metrics.total_tasks += 1

                task_id = item.get("task_id")
                completion = item.get("completion", "")
                generated_tests = item.get("generated_tests", "")
                passed_designer = item.get("passed_designer_tests", False)
                attempts = item.get("attempts_used", 1)

                if task_id not in self.official_data:
                    logger.warning(f"Task ID {task_id} not found in official dataset.")
                    continue

                official_task = self.official_data[task_id]

                # 1. Syntax Verification
                if self._check_syntax(completion):
                    self.metrics.syntax_correct += 1

                # 2. Test Validity
                if self._check_test_validity(generated_tests):
                    self.metrics.valid_tests_generated += 1

                # 3. Official Execution (True Pass@1)
                is_true_passed = self._run_sandbox(
                    completion, official_task["test"], official_task["entry_point"]
                )

                if is_true_passed:
                    self.metrics.true_passed += 1
                    self.metrics.total_attempts_for_success += attempts

                # 4. Designer False Positive Check
                if passed_designer:
                    self.metrics.designer_passed_total += 1
                    if not is_true_passed:
                        self.metrics.false_positives += 1

    def generate_report(self, output_dir: Path) -> Dict[str, Any]:
        """Calculates percentages and saves a reproducible report."""
        m = self.metrics
        total = max(m.total_tasks, 1)  # Prevent division by zero

        pass_at_1 = (m.true_passed / total) * 100
        syntax_rate = (m.syntax_correct / total) * 100
        test_validity_rate = (m.valid_tests_generated / total) * 100

        fp_rate = 0.0
        if m.designer_passed_total > 0:
            fp_rate = (m.false_positives / m.designer_passed_total) * 100

        avg_attempts = 0.0
        if m.true_passed > 0:
            avg_attempts = m.total_attempts_for_success / m.true_passed

        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics_raw": asdict(m),
            "metrics_calculated": {
                "true_pass_at_1_percent": round(pass_at_1, 2),
                "syntax_correct_percent": round(syntax_rate, 2),
                "test_validity_percent": round(test_validity_rate, 2),
                "designer_false_positive_percent": round(fp_rate, 2),
                "avg_attempts_for_success": round(avg_attempts, 2),
            },
        }

        # Print to console
        logger.info("\n" + "=" * 50)
        logger.info("🏆 EVALUATION REPORT")
        logger.info("=" * 50)
        logger.info(f"Total Evaluated:        {m.total_tasks}")
        logger.info(
            f"True Pass@1:            {pass_at_1:.2f}% ({m.true_passed}/{m.total_tasks})"
        )
        logger.info(f"Syntax Correctness:     {syntax_rate:.2f}%")
        logger.info(f"Test Gen Validity:      {test_validity_rate:.2f}%")
        logger.info(f"Designer False PosRate: {fp_rate:.2f}%")
        logger.info(f"Avg Attempts (Success): {avg_attempts:.2f}")
        logger.info("=" * 50)

        # Save to disk
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = (
            output_dir / f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        logger.info(f"Detailed report saved to: {report_path}")
        return report


# ============================================================================
# 3. CLI Entry Point
# ============================================================================
def main():

    parser = argparse.ArgumentParser(description="Academic Evaluator for AgentCoder.")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the JSONL results file generated by main.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save the evaluation report.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Timeout in seconds for each sandbox execution.",
    )

    args = parser.parse_args()

    evaluator = HumanEvalEvaluator(timeout=args.timeout)

    try:
        evaluator.evaluate_file(Path(args.input_path))
        evaluator.generate_report(Path(args.output_dir))
    except Exception as e:
        logger.error(f"Evaluation halted due to error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
