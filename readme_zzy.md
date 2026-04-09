~~~markdown
# Multi-Agent Project for CSIT5520, HKUST

A multi-agent collaboration project using Large Language Models (LLMs) for automated code generation, test generation, and execution feedback, developed for the CSIT5520 NLP course project at HKUST.

This project is inspired by the AgentCoder idea of separating the workflow into specialized agents:

- **Programmer Agent**: generates code for HumanEval tasks
- **Test Designer Agent**: generates independent test cases from the task prompt
- **Test Executor Agent**: executes generated code against generated tests and returns feedback

At the current stage, the repository mainly contains:

- a baseline code generation pipeline
- the implementation of the **Test Designer Agent**
- scripts for checking and refilling generated tests
- reproducible intermediate artifacts for evaluation and analysis

The current benchmark used in this project is **HumanEval**.

---

## 1. Repository Structure

```text
project/
├── data/
│   ├── baseline_samples.jsonl
│   └── designer_tests.jsonl
│
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   └── test_designer.py
│   │
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── designer_prompt.py
│   │
│   ├── check_designer_tests.py
│   ├── debug_empty_designer_tests.py
│   ├── generate_designer_tests.py
│   ├── refill_empty_designer_tests.py
│   └── run_baseline.py
│
├── eval.py
├── requirements.txt
├── .gitignore
└── README.md
~~~

------

## 2. Environment Setup

It is recommended to use a virtual environment.

```bash
# 1. Create a virtual environment
python3 -m venv venv

# 2. Activate the environment
# macOS / Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

------

## 3. API Setup

Create a `.env` file in the project root directory with your Azure OpenAI credentials.

```bash
AZURE_OPENAI_API_KEY=YOUR_API_KEY
AZURE_OPENAI_ENDPOINT="https://hkust.azure-api.net/"
AZURE_OPENAI_API_VERSION="2025-02-01-preview"
AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-5-mini"
```

Make sure the endpoint, API version, and deployment name match your Azure OpenAI allocation.

------

## 4. Dataset

This project currently uses the **HumanEval** benchmark loaded from Hugging Face.

- Dataset: `openai_humaneval`
- Split: `test`

------

## 5. Baseline Code Generation

The baseline uses a single LLM with zero-shot prompting to generate Python code completions for HumanEval tasks.

Run:

```bash
cd src
python run_baseline.py
```

Output:

```text
../data/baseline_samples.jsonl
```

------

## 6. Baseline Evaluation

To evaluate the baseline with Pass@1, run:

```bash
python eval.py
```

This script loads HumanEval, reads `data/baseline_samples.jsonl`, reconstructs full code from prompt + completion, and evaluates the results with Hugging Face `code_eval`.

------

## 7. Test Designer Agent

The **Test Designer Agent** generates independent test cases for each HumanEval task.

### Design Goal

The Test Designer does **not** look at the generated implementation.
Instead, it only uses:

- the task prompt
- function signature
- docstring
- examples in the prompt

This is intended to reduce test bias and keep test generation more objective.

### Prompt Design

The agent is prompted to generate:

- **Basic Cases**
- **Edge Cases**
- **Large Scale Cases**

At the same time, the prompt encourages:

- conservative test oracles
- high-confidence assertions
- avoiding risky assumptions
- avoiding undefined behavior when the specification is ambiguous

### Core Files

- `src/agents/test_designer.py`
- `src/prompts/designer_prompt.py`

------

## 8. Generate Test Designer Outputs

To generate tests for the full HumanEval set:

```bash
cd src
python generate_designer_tests.py
```

Output:

```text
../data/designer_tests.jsonl
```

Each line has the format:

```json
{"task_id": "HumanEval/0", "tests": "assert ..."}
```

This file is the main output artifact of the Test Designer component.

------

## 9. Check Empty Test Entries

Because LLM outputs may occasionally fail or return empty content, we include a checking script:

```bash
cd src
python check_designer_tests.py
```

This script scans `designer_tests.jsonl` and reports which tasks still have empty `tests` fields.

------

## 10. Refill Empty Test Entries

If some tasks failed to generate tests, we can re-run only those failed entries instead of regenerating the entire dataset.

Run:

```bash
cd src
python refill_empty_designer_tests.py
```

This script:

1. reads the existing `designer_tests.jsonl`
2. finds entries with empty `tests`
3. reloads the corresponding HumanEval tasks
4. re-runs the Test Designer only for those failed tasks
5. overwrites the file with updated results

------

## 11. Debugging Hard Failure Cases

For persistent failures, we provide a debugging script:

```bash
cd src
python debug_empty_designer_tests.py
```

This script is used to inspect hard cases individually and prints:

- finish reason
- raw message content
- retry behavior
- cleaned test output

It is intended for debugging and error analysis rather than the formal pipeline.

------

## 12. Current Status of Test Designer

After full generation, checking, and refill:

- most HumanEval tasks successfully obtained generated tests
- some tasks initially had empty outputs
- most of those failures were recovered after refill
- **3 HumanEval tasks still remain empty**:
  - `HumanEval/32`
  - `HumanEval/81`
  - `HumanEval/145`

We keep these remaining cases as **failure samples** for error analysis instead of manually writing replacement tests.

This reflects a practical limitation of LLM-based test generation: most tasks can be handled successfully, but a small number of hard cases may still fail because of unstable or empty model outputs.

------

## 13. Main Output Files

### `data/baseline_samples.jsonl`

Baseline code completions generated by the zero-shot baseline.

### `data/designer_tests.jsonl`

Independent test cases generated by the Test Designer Agent for HumanEval.

These are the main reproducible artifacts currently kept in the repository.

------

## 14. Recommended Workflow

### Baseline

```bash
cd src
python run_baseline.py
cd ..
python eval.py
```

### Test Designer

```bash
cd src
python generate_designer_tests.py
python check_designer_tests.py
python refill_empty_designer_tests.py
python check_designer_tests.py
```

### Optional Debugging

```bash
python debug_empty_designer_tests.py
```

------

## 15. Current Focus and Future Work

The current repository mainly focuses on the baseline and Test Designer components.

Planned next steps include:

- implementing the Test Executor Agent
- combining programmer output and designer tests into executable feedback loops
- building the complete multi-agent pipeline
- comparing multi-agent results against the baseline
- conducting ablation studies and error analysis in the final report

------

## 16. Course Project Context

This repository is part of the CSIT5520 course project on **Multi-Agent Collaboration**.

The project studies whether multiple specialized LLM agents can improve the reliability of automated code generation through collaboration, testing, and execution feedback.
