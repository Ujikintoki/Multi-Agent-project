# src/agents/programmer.py
import re
from typing import Optional

from src.agents.llm_client import AsyncLLMClient
from src.prompts.programmer_prompt import (
    PROGRAMMER_REFINE_PROMPT,
    PROGRAMMER_SYSTEM_PROMPT,
)


class ProgrammerAgent:
    def __init__(self, llm_client: AsyncLLMClient):
        self.llm_client = llm_client

    def _extract_code(self, raw_text: str) -> str:
        """Utility method to strip markdown tags and extract pure Python code."""
        if not raw_text:
            return ""
        # Match anything between ```python and ```
        pattern = r"```python\s*(.*?)\s*```"
        match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Fallback if the LLM forgets the markdown tags
        return raw_text.strip()

    async def generate_initial_code(self, problem_description: str) -> Optional[str]:
        """Generates the first pass of the code based on the HumanEval prompt."""
        messages = [
            {"role": "system", "content": PROGRAMMER_SYSTEM_PROMPT},
            {"role": "user", "content": problem_description},
        ]

        raw_response = await self.llm_client.generate_response(
            messages, temperature=1.0
        )
        return self._extract_code(raw_response)

    async def refine_code(
        self, problem_description: str, original_code: str, error_feedback: str
    ) -> Optional[str]:
        """Refines the code based on stdout/stderr from the Executor."""
        user_content = (
            f"--- PROBLEM ---\n{problem_description}\n\n"
            f"--- YOUR PREVIOUS CODE ---\n{original_code}\n\n"
            f"--- EXECUTION ERROR / TEST FAILURE ---\n{error_feedback}\n"
        )

        messages = [
            {"role": "system", "content": PROGRAMMER_REFINE_PROMPT},
            {"role": "user", "content": user_content},
        ]

        raw_response = await self.llm_client.generate_response(
            messages, temperature=1.0
        )
        return self._extract_code(raw_response)
