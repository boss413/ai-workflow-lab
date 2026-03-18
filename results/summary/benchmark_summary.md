# LLM Benchmark Results

Generated: 2026-03-18 11:35:17  
Tasks: classification, extraction, generation, qa, reasoning, summarization  
Total result files: 12

---

## Classification

| Model                                                | Tier   | N   | Runs | Accuracy | Avg Conf | Latency(s) | Cost($) |
| ---------------------------------------------------- | ------ | --- | ---- | -------- | -------- | ---------- | ------- |
| gemini_pro (google:gemini-2.5-pro)                   | medium | 100 | 2    | 86%      | -        | 6.83s      | $0.0152 |
| claude_sonnet (anthropic:claude-sonnet-4-5-20250929) | medium | 100 | 2    | 85%      | 94%      | 1.65s      | $0.0587 |
| gpt41 (openai:gpt-4.1)                               | medium | 100 | 2    | 84%      | 99%      | 0.69s      | $0.0228 |
| gemini_flash (google:gemini-2.5-flash)               | low    | 100 | 2    | 80%      | -        | 1.85s      | $0.0006 |
| gpt41_mini (openai:gpt-4.1-mini)                     | low    | 100 | 2    | 79%      | 98%      | 0.71s      | $0.0045 |
| claude_haiku (anthropic:claude-haiku-4-5-20251001)   | low    | 100 | 2    | 79%      | 87%      | 1.01s      | $0.0195 |
| mistral_7b (ollama:mistral:latest)                   | low    | 100 | 2    | 69%      | -        | 3.17s      | $0.0000 |

## Extraction

| Model                                                | Tier   | N   | Runs | Accuracy | F1  | Precision | Recall | Latency(s) | Cost($) |
| ---------------------------------------------------- | ------ | --- | ---- | -------- | --- | --------- | ------ | ---------- | ------- |
| claude_sonnet (anthropic:claude-sonnet-4-5-20250929) | medium | 100 | 2    | 94%      | 87% | 86%       | 90%    | 1.70s      | $0.1034 |
| gemini_flash (google:gemini-2.5-flash)               | low    | 100 | 2    | 87%      | 83% | 82%       | 84%    | 1.45s      | $0.0051 |
| gpt41 (openai:gpt-4.1)                               | medium | 100 | 2    | 86%      | 80% | 78%       | 82%    | 0.78s      | $0.0387 |
| gpt41_mini (openai:gpt-4.1-mini)                     | low    | 100 | 2    | 84%      | 79% | 78%       | 81%    | 1.01s      | $0.0078 |
| gemini_pro (google:gemini-2.5-pro)                   | medium | 100 | 2    | 85%      | 78% | 77%       | 82%    | 5.90s      | $0.0514 |
| claude_haiku (anthropic:claude-haiku-4-5-20251001)   | low    | 100 | 2    | 87%      | 77% | 74%       | 83%    | 0.89s      | $0.0347 |
| mistral_7b (ollama:mistral:latest)                   | low    | 100 | 2    | 76%      | 67% | 68%       | 67%    | 7.95s      | $0.0000 |

## Generation

| Model                                                | Tier   | N   | Runs | Accuracy | Coverage | ROUGE-L | Latency(s) | Cost($) |
| ---------------------------------------------------- | ------ | --- | ---- | -------- | -------- | ------- | ---------- | ------- |
| gpt41_mini (openai:gpt-4.1-mini)                     | low    | 100 | 2    | 14%      | 17%      | 19%     | 1.16s      | $0.0085 |
| mistral_7b (ollama:mistral:latest)                   | low    | 100 | 2    | 16%      | 17%      | 14%     | 3.98s      | $0.0000 |
| claude_haiku (anthropic:claude-haiku-4-5-20251001)   | low    | 100 | 2    | 11%      | 12%      | 17%     | 1.40s      | $0.0350 |
| gpt41 (openai:gpt-4.1)                               | medium | 100 | 2    | 6%       | 6%       | 20%     | 1.36s      | $0.0448 |
| claude_sonnet (anthropic:claude-sonnet-4-5-20250929) | medium | 100 | 2    | 5%       | 5%       | 19%     | 2.87s      | $0.1130 |
| gemini_pro (google:gemini-2.5-pro)                   | medium | 100 | 2    | 4%       | 4%       | 20%     | 15.30s     | $0.0436 |
| gemini_flash (google:gemini-2.5-flash)               | low    | 100 | 2    | 2%       | 2%       | 21%     | 4.00s      | $0.0038 |

## Qa

| Model                                                | Tier   | N   | Runs | Accuracy | F1  | Exact Match | Latency(s) | Cost($) |
| ---------------------------------------------------- | ------ | --- | ---- | -------- | --- | ----------- | ---------- | ------- |
| gemini_pro (google:gemini-2.5-pro)                   | medium | 100 | 2    | 82%      | 81% | 62%         | 3.92s      | $0.0406 |
| mistral_7b (ollama:mistral:latest)                   | low    | 100 | 2    | 54%      | 58% | 31%         | 5.76s      | $0.0000 |
| gemini_flash (google:gemini-2.5-flash)               | low    | 100 | 2    | 51%      | 56% | 28%         | 1.27s      | $0.0031 |
| gpt41 (openai:gpt-4.1)                               | medium | 100 | 2    | 51%      | 55% | 26%         | 0.82s      | $0.0595 |
| claude_sonnet (anthropic:claude-sonnet-4-5-20250929) | medium | 100 | 2    | 28%      | 41% | 12%         | 1.68s      | $0.1368 |
| gpt41_mini (openai:gpt-4.1-mini)                     | low    | 100 | 2    | 28%      | 38% | 1%          | 0.90s      | $0.0126 |
| claude_haiku (anthropic:claude-haiku-4-5-20251001)   | low    | 100 | 2    | 20%      | 33% | 2%          | 0.88s      | $0.0497 |

## Reasoning

| Model                                                | Tier   | N   | Runs | Accuracy | Parse Errs | Avg Tokens | Latency(s) | Cost($) |
| ---------------------------------------------------- | ------ | --- | ---- | -------- | ---------- | ---------- | ---------- | ------- |
| claude_sonnet (anthropic:claude-sonnet-4-5-20250929) | medium | 100 | 2    | 98%      | 0          | 246        | 4.09s      | $0.4235 |
| claude_haiku (anthropic:claude-haiku-4-5-20251001)   | low    | 100 | 2    | 95%      | 0          | 229        | 2.00s      | $0.1317 |
| gpt41 (openai:gpt-4.1)                               | medium | 100 | 2    | 95%      | 0          | 182        | 2.56s      | $0.1761 |
| gpt41_mini (openai:gpt-4.1-mini)                     | low    | 100 | 2    | 94%      | 0          | 180        | 3.04s      | $0.0342 |
| gemini_pro (google:gemini-2.5-pro)                   | medium | 100 | 2    | 92%      | 0          | 248        | 10.59s     | $0.2744 |
| gemini_flash (google:gemini-2.5-flash)               | low    | 97  | 2    | 71%      | 1          | 191        | 2.44s      | $0.0264 |
| mistral_7b (ollama:mistral:latest)                   | low    | 100 | 2    | 57%      | 0          | 253        | 9.11s      | $0.0000 |

## Summarization

| Model                                                | Tier   | N   | Runs | Accuracy | ROUGE-1 | ROUGE-2 | ROUGE-L | Latency(s) | Cost($) |
| ---------------------------------------------------- | ------ | --- | ---- | -------- | ------- | ------- | ------- | ---------- | ------- |
| gemini_pro (google:gemini-2.5-pro)                   | medium | 100 | 2    | 70%      | 24%     | 7%      | 16%     | 12.58s     | $0.1495 |
| gemini_flash (google:gemini-2.5-flash)               | low    | 100 | 2    | 61%      | 23%     | 5%      | 17%     | 3.17s      | $0.0094 |
| claude_sonnet (anthropic:claude-sonnet-4-5-20250929) | medium | 100 | 2    | 57%      | 20%     | 5%      | 14%     | 3.91s      | $0.3916 |
| gpt41 (openai:gpt-4.1)                               | medium | 100 | 2    | 51%      | 20%     | 5%      | 14%     | 1.79s      | $0.1928 |
| gpt41_mini (openai:gpt-4.1-mini)                     | low    | 100 | 2    | 46%      | 19%     | 4%      | 13%     | 2.05s      | $0.0385 |
| claude_haiku (anthropic:claude-haiku-4-5-20251001)   | low    | 100 | 2    | 39%      | 18%     | 4%      | 13%     | 2.09s      | $0.1325 |
| mistral_7b (ollama:mistral:latest)                   | low    | 100 | 2    | 38%      | 18%     | 4%      | 12%     | 27.94s     | $0.0000 |
