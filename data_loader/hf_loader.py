from datasets import load_dataset
from .dataset_registry import DATASET_REGISTRY


def load_task_dataset(task_name, sample_size=None):

    config = DATASET_REGISTRY[task_name]

    dataset = load_dataset(
        config["hf_name"],
        split=config["split"]
    )

    if sample_size:
        dataset = dataset.select(range(sample_size))

    normalized = []

    for item in dataset:
        normalized.append({
            "input": item[config["input_field"]],
            "label": item[config["label_field"]],
            "metadata": {}
        })

    return normalized