import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DESIGN_DOC = ROOT / "docs/design_doc.md"
SPEC_DOC = ROOT / "docs/implementation_spec.md"
CODING_RULES = ROOT / "standards/coding_rules.md"


def read_file(path):
    if not path.exists():
        return ""
    return path.read_text()


def compile_prompt(task_file):

    task_path = ROOT / "tasks" / task_file

    if not task_path.exists():
        raise FileNotFoundError(f"Task file not found: {task_path}")

    design = read_file(DESIGN_DOC)
    spec = read_file(SPEC_DOC)
    rules = read_file(CODING_RULES)
    task = read_file(task_path)

    prompt = f"""
You are a senior Python engineer working on an AI benchmarking system.

Follow the project design and coding standards when implementing the task.

========================
PROJECT DESIGN
========================
{design}

========================
IMPLEMENTATION SPEC
========================
{spec}

========================
CODING RULES
========================
{rules}

========================
TASK
========================
{task}

========================
OUTPUT REQUIREMENTS
========================
Return complete Python code and tests.
Do not include explanations.
"""

    return prompt


def main():

    if len(sys.argv) != 2:
        print("Usage: python compile_task_prompt.py <task_file>")
        return

    task_file = sys.argv[1]

    prompt = compile_prompt(task_file)

    print(prompt)


if __name__ == "__main__":
    main()