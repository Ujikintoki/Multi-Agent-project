import os
import json
from datasets import load_dataset
import evaluate

# 关键设置：允许 Hugging Face 在本地执行代码（为了验证代码是否能跑通）
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

def run_hf_eval():
    print("Loading Hugging Face code_eval metric...")
    # 1. 加载 HF 官方代码评估指标
    code_eval = evaluate.load("code_eval")
    
    print("Loading HumanEval dataset...")
    # 2. 加载原数据集以获取 Prompt 和 Test Case
    dataset = load_dataset("openai_humaneval", split="test")
    
    # 3. 读取你生成的答案 (确保路径正确，如果脚本和 jsonl 都在 src 下，可以直接填文件名)
    completions = {}
    jsonl_path = "../data/baseline_samples.jsonl" 
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            completions[data["task_id"]] = data["completion"]
            
    # 4. 组装评测数据
    predictions = []
    references = []
    
    for problem in dataset:
        task_id = problem["task_id"]
        # 完整的代码 = 原始 prompt + 模型生成的补全
        full_code = problem["prompt"] + completions.get(task_id, "")
        # HF code_eval 要求 predictions 是二维列表 [[code1, code2...]] 以支持 Pass@k
        predictions.append([full_code])  
        
        # 测试用例 = 官方提供的 test 字符串 + 显式调用 entry_point 检查函数
        test_case = problem["test"] + f"\ncheck({problem['entry_point']})\n"
        references.append(test_case)
        
    print(f"Starting evaluation for {len(predictions)} problems. This may take a moment...")
    # 5. 计算 Pass@1
    results, _ = code_eval.compute(references=references, predictions=predictions, k=[1])
    
    print("\n==================================")
    print("Baseline 评估结果 (Pass@1):")
    print(results)
    print("==================================")

if __name__ == "__main__":
    run_hf_eval()