# Task 10: Evaluation Metrics

## Objective
Score benchmark outputs.

## Files
evaluation/metrics.py

## Interface

score(task: str, prediction: str, truth: dict)

## Metrics

classification → accuracy  
extraction → schema match  
summarization → ROUGE  
QA → exact match  
reasoning → answer match  
generation → rubric

## Tests
tests/test_metrics.py