import os
import json
import asyncio
from dotenv import load_dotenv
from datasets import load_dataset
from openai import AsyncAzureOpenAI
from tqdm.asyncio import tqdm

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT").rstrip('/')
deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

azure_client = AsyncAzureOpenAI(
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
    default_headers={"Ocp-Apim-Subscription-Key": api_key},
    timeout=60.0,
)

async def generate_code(task_id: str, prompt: str) -> dict:
    """调用大模型生成代码补全"""
    try:
        # Zero-shot prompting 建立 Baseline
        response = await azure_client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are an expert Python programmer. Complete the given Python function. ONLY return the valid Python code. Do not wrap it in markdown block quotes (like ```python) and do not provide explanations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512
        )
        
        # 提取生成的代码，并做简单的清理（移除可能带有的 markdown 标记）
        generated_text = response.choices[0].message.content
        clean_code = generated_text.replace("```python", "").replace("```", "").strip()
        
        return {
            "task_id": task_id,
            "completion": clean_code
        }
    except Exception as e:
        print(f"Error generating code for {task_id}: {e}")
        return {
            "task_id": task_id,
            "completion": ""
        }

async def main():
    print("Loading HumanEval dataset...")
    # 下载并加载标准的 HumanEval 测试集
    dataset = load_dataset("openai_humaneval")
    problems = dataset["test"]
    
    print(f"Loaded {len(problems)} problems. Starting generation...")
    
    # 并发请求 API（为了避免触发速率限制，可以适当分批）
    tasks = [generate_code(problem["task_id"], problem["prompt"]) for problem in problems]
    results = await tqdm.gather(*tasks)
    
    # 保存为官方评测脚本所需的 JSONL 格式
    output_file = "../data/baseline_samples.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
            
    print(f"\nGeneration complete! Baseline samples saved to {output_file}.")
    print("Next step: Use the 'human-eval' library to calculate Pass@1.")

if __name__ == "__main__":
    asyncio.run(main())