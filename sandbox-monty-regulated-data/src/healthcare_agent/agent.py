"""AI agent: generates code via Claude, executes in Monty sandbox."""

import os
import re

import anthropic
import pydantic_monty

from .prompts import SYSTEM_PROMPT

DEFAULT_MODEL = "claude-opus-4-6"


def get_model() -> str:
    return os.environ.get("CLAUDE_MODEL", DEFAULT_MODEL)


def generate_code(step_name: str, prompt: str) -> str:
    """Call Claude to generate Python code for a pipeline step.

    Model is read from CLAUDE_MODEL env var (default: claude-opus-4-6).
    Returns clean Python code string (markdown fences stripped).
    """
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=get_model(),
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    code = response.content[0].text
    # Strip markdown code fences if present
    code = re.sub(r"^```(?:python)?\n?", "", code, flags=re.MULTILINE)
    code = re.sub(r"\n?```$", "", code, flags=re.MULTILINE)
    return code.strip()


def execute_in_sandbox(code: str, records: list) -> dict:
    """Execute AI-generated code in Monty sandbox.

    The code must define and call process(records) → dict.
    No filesystem, no network, no env vars — pure computation.
    """
    m = pydantic_monty.Monty(code, inputs=["records"])
    return m.run(inputs={"records": records})
