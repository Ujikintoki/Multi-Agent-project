# src/check_designer_tests.py
import json

path = "../data/designer_tests.jsonl"
empty_ids = []

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if not item.get("tests", "").strip():
            empty_ids.append(item["task_id"])

print("Empty test entries:", len(empty_ids))
print(empty_ids)