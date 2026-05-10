import requests
import json
import os
import csv
from typing import List, Dict

API_URL = "http://localhost:8000/query"

test_questions = [
    {"question": "What is the target completion date for Project Phoenix Phase 1?", "expected_answer": "Q4 2026"},
    {"question": "How many days of vacation do full-time employees get?", "expected_answer": "20 days"},
    {"question": "What database is Project Phoenix migrating to?", "expected_answer": "PostgreSQL"},
    {"question": "How long is the paternity leave?", "expected_answer": "8 weeks"}
]

def run_eval(output_csv: str = "eval_results.csv"):
    results = []
    
    for q in test_questions:
        question = q["question"]
        print(f"Testing: {question}")
        
        try:
            response = requests.post(API_URL, json={"query": question})
            if response.status_code == 200:
                data = response.json()
                results.append({
                    "question": question,
                    "expected_answer": q["expected_answer"],
                    "generated_answer": data.get("answer", ""),
                    "sources": " | ".join(data.get("sources", []))
                })
            else:
                results.append({
                    "question": question,
                    "expected_answer": q["expected_answer"],
                    "generated_answer": f"ERROR: {response.text}",
                    "sources": ""
                })
        except Exception as e:
            results.append({
                "question": question,
                "expected_answer": q["expected_answer"],
                "generated_answer": f"EXCEPTION: {e}",
                "sources": ""
            })
            
    # Write to CSV
    keys = results[0].keys() if results else []
    with open(output_csv, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
        
    print(f"Evaluation complete. Results saved to {output_csv}")

if __name__ == "__main__":
    print("Ensure the FastAPI server is running on http://localhost:8000")
    run_eval()
