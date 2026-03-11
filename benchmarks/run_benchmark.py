import time
import json
from datasets import load_dataset

from models.openai_model import run_openai
from models.anthropic_model import run_anthropic
from evaluation.metrics import score

MODELS = {
    "openai": run_openai,
    "anthropic": run_anthropic
}

def run_benchmark(task_name, dataset_name, prompt_template, n=100):

    dataset = load_dataset(dataset_name, split=f"train[:{n}]")

    results = []

    for row in dataset:

        for model_name, model_fn in MODELS.items():

            prompt = prompt_template(row)

            start = time.time()
            response, usage = model_fn(prompt)
            latency = time.time() - start

            score_result = score(task_name, response, row)

            record = {
                "task": task_name,
                "model": model_name,
                "correct": score_result["correct"],
                "latency": latency,
                "tokens_in": usage["input"],
                "tokens_out": usage["output"],
                "cost": usage["cost"]
            }

            results.append(record)

    with open(f"results/raw/{task_name}.json", "w") as f:
        json.dump(results, f)

    return results