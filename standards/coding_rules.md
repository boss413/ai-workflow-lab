# Coding Standards

These rules apply to all code generated for this project.

## Language

Python 3.11+

Use modern Python features and type hints.

---

## Code Style

- Follow PEP8 formatting.
- Use descriptive variable names.
- Functions should be under 50 lines where possible.
- Avoid deeply nested logic.

---

## Typing

All public functions must include type hints.

Example:

def load_config() -> dict:

---

## Error Handling

Do not allow silent failures.

Raise explicit errors when:

- configuration is invalid
- required keys are missing
- files cannot be loaded

---

## Dependencies

Only use standard library unless explicitly required.

Approved libraries:

- pyyaml
- datasets
- pandas
- pytest

---

## Testing

Use pytest.

Every module must have tests.

Tests should verify:

- correct outputs
- failure conditions
- edge cases

---

## File Organization

One module per responsibility.

Example:

datasets/
models/
evaluation/

Avoid large monolithic files.

---

## Logging

Use Python logging module instead of print statements.

---

## Security

Never hardcode:

- API keys
- tokens
- credentials

Secrets must be loaded from environment variables.

---

## Output Requirements

Generated code must:

- run without modification
- include imports
- include tests