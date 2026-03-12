from data_loader.hf_loader import load_task_dataset

data = load_task_dataset("classification", sample_size=5)

for row in data:
    print(row)