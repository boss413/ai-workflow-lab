import pandas as pd

def summarize(results):

    df = pd.DataFrame(results)

    summary = df.groupby(["model", "task"]).agg({
        "correct": "mean",
        "latency": "mean",
        "cost": "mean"
    })

    return summary