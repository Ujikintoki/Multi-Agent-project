import os
import re
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

from prompts.designer_prompt import (
    DESIGNER_SYSTEM_PROMPT,
    build_designer_user_prompt,
)

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")


class TestDesignerAgent:
    def __init__(self):
        if not api_key:
            raise ValueError("Missing AZURE_OPENAI_API_KEY in .env")
        if not api_version:
            raise ValueError("Missing AZURE_OPENAI_API_VERSION in .env")
        if not azure_endpoint:
            raise ValueError("Missing AZURE_OPENAI_ENDPOINT in .env")
        if not deployment_name:
            raise ValueError("Missing AZURE_OPENAI_CHAT_DEPLOYMENT in .env")

        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            default_headers={"Ocp-Apim-Subscription-Key": api_key},
            timeout=60.0,
        )
        self.model = deployment_name

    async def generate_tests(self, task_id: str, problem_prompt: str) -> dict:
        """
        Generate test cases for one HumanEval task.

        Returns:
            {
                "task_id": ...,
                "tests": ...
            }
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": DESIGNER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_designer_user_prompt(problem_prompt),
                    },
                ],
                max_tokens=700,
                temperature=0.2,
            )

            raw_text = response.choices[0].message.content or ""
            cleaned_tests = self._clean_output(raw_text)

            return {
                "task_id": task_id,
                "tests": cleaned_tests,
            }

        except Exception as e:
            print(f"Error generating tests for {task_id}: {e}")
            return {
                "task_id": task_id,
                "tests": "",
            }

    def _clean_output(self, text: str) -> str:
        """
        Clean model output so it is easier for executor to run.
        """
        text = text.strip()

        # Remove markdown fences if the model still returns them
        text = text.replace("```python", "")
        text = text.replace("```py", "")
        text = text.replace("```", "")
        text = text.strip()

        # Remove accidental leading phrases before first useful line
        lines = text.splitlines()
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()

            if not stripped:
                cleaned_lines.append(line)
                continue

            # Keep comments
            if stripped.startswith("#"):
                cleaned_lines.append(line)
                continue

            # Keep assert lines
            if stripped.startswith("assert "):
                cleaned_lines.append(line)
                continue

            # Keep simple helper/test data definitions
            if self._is_allowed_code_line(stripped):
                cleaned_lines.append(line)
                continue

            # Otherwise drop obvious natural language lines
            # Example: "Here are the test cases:"
            if not self._looks_like_code(stripped):
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    def _is_allowed_code_line(self, line: str) -> bool:
        allowed_starts = (
            "for ",
            "if ",
            "elif ",
            "else:",
            "while ",
            "try:",
            "except",
            "large_",
            "data_",
            "test_",
            "cases_",
            "sample_",
            "values_",
            "nums_",
            "arr_",
            "lst_",
            "input_",
            "inputs_",
            "expected_",
            "result_",
            "results_",
            "from ",
            "import ",
        )

        if line.startswith(allowed_starts):
            return True

        # Allow simple assignments like:
        # large_input = [...]
        # nums = [1, 2, 3]
        if "=" in line and not line.startswith(("def ", "class ")):
            return True

        return False

    def _looks_like_code(self, line: str) -> bool:
        """
        Heuristic: decide whether a line looks like code.
        """
        code_patterns = [
            r"^\w+\s*=",
            r"^assert\s+",
            r"^#",
            r"^for\s+",
            r"^if\s+",
            r"^while\s+",
            r"^from\s+\w+",
            r"^import\s+\w+",
        ]

        for pattern in code_patterns:
            if re.match(pattern, line):
                return True
        return False
