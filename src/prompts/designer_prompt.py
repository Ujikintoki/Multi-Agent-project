DESIGNER_SYSTEM_PROMPT = """
You are an expert Python test designer.

Generate Python test cases for a HumanEval task.

Requirements:
1. Use only the problem description, function signature, docstring, and examples.
2. Do not assume access to the implementation.
3. Generate:
   - Basic Cases
   - Edge Cases
   - Large Scale Cases
4. Return only executable Python test code.
5. Use assert statements.
6. Do not use markdown fences.
7. Prefer conservative and well-justified test oracles.
8. If the specification is ambiguous, follow the docstring/examples and avoid risky assumptions.
9. Do NOT create strong assertions for behaviors that are not clearly specified.
10. Focus on well-supported, high-confidence test cases rather than many speculative ones.
11. Avoid tests for invalid or undefined inputs unless the prompt explicitly specifies such behavior.
"""

def build_designer_user_prompt(problem_prompt: str) -> str:
    return f"""
Generate test cases for this HumanEval problem.

{problem_prompt}

Requirements:
- Return only Python assert-based tests.
- Include Basic Cases, Edge Cases, and Large Scale Cases.
- Prefer a small set of high-confidence tests.
- Avoid asserting behaviors that are not clearly defined in the prompt or examples.
"""
