DATASET_REGISTRY = {

    "classification": {
        "hf_name": "ag_news",
        "input_field": "text",
        "label_field": "label",
        "split": "test"
    },

    "summarization": {
        "hf_name": "xsum",
        "input_field": "document",
        "label_field": "summary",
        "split": "test"
    },

    "qa": {
        "hf_name": "natural_questions",
        "input_field": "question",
        "label_field": "answer",
        "split": "validation"
    }

}