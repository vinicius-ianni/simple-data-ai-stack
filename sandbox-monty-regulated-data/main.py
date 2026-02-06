"""Healthcare AI Agent — Secure Data Pipeline.

Run: ANTHROPIC_API_KEY=your-key uv run python main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from healthcare_agent.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()
