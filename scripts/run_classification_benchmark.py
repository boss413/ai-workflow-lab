from data_loader.hf_loader import load_task_dataset
from dotenv import load_dotenv
import time, json
from models.openai_adapter import OpenAIAdapter

load_dotenv()

dataset = load_task_dataset("classification", sample_size=20)
adapter = OpenAIAdapter(model_name="gpt-4.1-mini")
results = []

for row in dataset:
    response = adapter.run(row["input"])
    predicted_label = response.text.strip()
    correct = int(predicted_label == row["label"])
    results.append({
        "input": row["input"],
        "predicted": predicted_label,
        "actual": row["label"],
        "correct": correct,
        "latency": response.latency,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "cost": response.cost,
    })

with open("results/classification_sample.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Done. {sum(r['correct'] for r in results)}/{len(results)} correct.")