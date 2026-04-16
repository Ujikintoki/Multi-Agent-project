import os
import re
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

from src.prompts.designer_prompt import (
    DESIGNER_SYSTEM_PROMPT,
    build_designer_user_prompt,
)

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")


class TestDesignerAgent:
    def __init__(self, debug: bool = False):
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
        self.debug = debug

    async def generate_tests(self, task_id: str, problem_prompt: str) -> dict:
        """
        Generate test cases for one HumanEval task.

        Returns:
            {
                "task_id": "...",
                "tests": "..."
            }
        """
        try:
            # First attempt: normal prompt
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": DESIGNER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_designer_user_prompt(problem_prompt),
                    },
                ],
                max_tokens=2000,
            )

            choice = response.choices[0]
            message = choice.message

            if self.debug:
                print("===== FINISH REASON =====")
                print(choice.finish_reason)
                print("===== END FINISH REASON =====")

                print("===== RAW MESSAGE CONTENT =====")
                print(message.content)
                print("===== END RAW MESSAGE CONTENT =====")

            raw_text = self._extract_text_from_message_content(message.content)

            # Retry once if the model got cut off before producing visible content
            if choice.finish_reason == "length" and not raw_text.strip():
                if self.debug:
                    print("===== RETRYING WITH SHORTER PROMPT =====")

                short_user_prompt = (
                    f"Generate a small set of high-confidence Python assert tests for this "
                    f"HumanEval problem.\n\n"
                    f"{problem_prompt}\n\n"
                    f"Requirements:\n"
                    f"- Return only Python test code\n"
                    f"- Use assert statements\n"
                    f"- No explanations\n"
                    f"- No markdown fences\n"
                    f"- Prefer fewer but reliable tests\n"
                )

                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": DESIGNER_SYSTEM_PROMPT},
                        {"role": "user", "content": short_user_prompt},
                    ],
                    max_tokens=2000,
                )

                choice = response.choices[0]
                message = choice.message

                if self.debug:
                    print("===== RETRY FINISH REASON =====")
                    print(choice.finish_reason)
                    print("===== END RETRY FINISH REASON =====")

                    print("===== RETRY RAW MESSAGE CONTENT =====")
                    print(message.content)
                    print("===== END RETRY RAW MESSAGE CONTENT =====")

                raw_text = self._extract_text_from_message_content(message.content)

            if self.debug:
                print("===== RAW MODEL OUTPUT =====")
                print(raw_text)
                print("===== END RAW MODEL OUTPUT =====")

            cleaned_tests = self._clean_output(raw_text)

            if self.debug:
                print("===== CLEANED TESTS =====")
                print(cleaned_tests)
                print("===== END CLEANED TESTS =====")

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

    def _extract_text_from_message_content(self, content) -> str:
        """
        Extract plain text from the model response content.
        Supports both plain string content and structured content lists.
        """
        if content is None:
            return ""

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts = []
            for part in content:
                # Dictionary-style content block
                if isinstance(part, dict):
                    if part.get("type") == "text" and "text" in part:
                        text_parts.append(part["text"])

                # Object-style content block
                else:
                    if getattr(part, "type", None) == "text":
                        text_value = getattr(part, "text", "")
                        if text_value:
                            text_parts.append(text_value)

            return "\n".join(text_parts).strip()

        return str(content).strip()

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

            # Keep helper/test data definitions
            if self._is_allowed_code_line(stripped):
                cleaned_lines.append(line)
                continue

            # Drop obvious natural language lines
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
