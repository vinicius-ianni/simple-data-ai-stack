"""Pipeline orchestration with Rich live terminal dashboard."""

import time
from datetime import datetime

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .agent import execute_in_sandbox, generate_code, get_model
from .data import FAKE_PATIENTS
from .prompts import STEP_PROMPTS

STEPS = [
    ("validate", "Validate Records", "Checking required fields & data integrity"),
    ("normalize", "Normalize Formats", "Standardizing dates, codes & lab units"),
    ("deidentify", "De-identify PHI", "Masking names, SSNs, addresses, DOBs"),
    ("enrich", "Enrich Records", "Adding glucose/cholesterol/A1C flags"),
    ("risk_score", "Compute Risk Scores", "Scoring patients 0-100 based on labs & diagnoses"),
    ("aggregate", "Aggregate Stats", "Computing diagnosis counts, risk distribution & lab averages"),
]

# Steps whose output replaces current_records for the next step
RECORD_PASSTHROUGH = {"validate": "valid", "normalize": "records", "deidentify": "records", "enrich": "records", "risk_score": "records"}

SPINNER_FRAMES = ["   ", ".  ", ".. ", "...", " ..", "  .", "   "]

console = Console()

NUM_STEPS = len(STEPS)


def _format_us(seconds: float) -> str:
    """Format duration with the right unit to highlight speed."""
    us = seconds * 1_000_000
    if us < 1000:
        return f"{us:.0f}us"
    ms = seconds * 1000
    if ms < 1000:
        return f"{ms:.1f}ms"
    return f"{seconds:.2f}s"


def _header_panel(pipeline_start: float | None = None, steps_done: int = 0) -> Panel:
    header = Text()
    header.append("\n")
    header.append("  HEALTHCARE AI AGENT  ", style="bold white on blue")
    header.append("  Secure Data Pipeline  ", style="bold white on dark_green")
    header.append("\n\n")
    header.append("  Engine    ", style="dim")
    header.append("Monty Sandbox", style="bold green")
    header.append("  (Rust-based secure Python interpreter)\n", style="dim")
    model = get_model()
    header.append("  AI Model  ", style="dim")
    header.append(model, style="bold yellow")
    header.append("  (code generation)\n", style="dim")
    header.append("  Security  ", style="dim")
    header.append("No filesystem ", style="bold red")
    header.append("| ", style="dim")
    header.append("No network ", style="bold red")
    header.append("| ", style="dim")
    header.append("No env vars ", style="bold red")
    header.append("| ", style="dim")
    header.append("No imports", style="bold red")
    header.append(f"\n  Progress  ", style="dim")
    bar_done = "█" * steps_done
    bar_remaining = "░" * (NUM_STEPS - steps_done)
    header.append(bar_done, style="bold green")
    header.append(bar_remaining, style="dim")
    header.append(f"  {steps_done}/{NUM_STEPS} steps", style="bold cyan")
    if pipeline_start is not None:
        elapsed = time.time() - pipeline_start
        header.append(f"  ({elapsed:.1f}s elapsed)", style="dim")
    header.append("\n")
    return Panel(header, border_style="cyan", title="[bold cyan]MONTY SANDBOX DEMO[/bold cyan]")


def _data_panel(records: list) -> Panel:
    table = Table(show_header=True, header_style="bold magenta", expand=True, show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Patient ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("DOB", style="white")
    table.add_column("SSN", style="red")
    table.add_column("Diagnosis", style="green")
    table.add_column("Status", justify="center")
    for i, r in enumerate(records, 1):
        name = r.get("name", "[red bold]MISSING[/red bold]")
        ssn = r.get("ssn", "[red bold]MISSING[/red bold]")
        dob = r.get("dob", "[red bold]MISSING[/red bold]")
        ok = all(k in r for k in ["patient_id", "name", "dob", "ssn", "diagnosis_code"])
        status = "[green bold]OK[/green bold]" if ok else "[red bold]INCOMPLETE[/red bold]"
        table.add_row(str(i), r["patient_id"], name, dob, ssn, r.get("diagnosis_code", "?"), status)
    return Panel(table, title=f"[magenta]Input Data: {len(records)} Patient Records (FAKE PHI)[/magenta]", border_style="magenta")


def _timing_bar(label: str, seconds: float, max_seconds: float, color: str) -> Text:
    """Render a visual timing bar."""
    bar_width = 30
    filled = int((seconds / max_seconds) * bar_width) if max_seconds > 0 else 0
    filled = max(1, min(filled, bar_width))
    bar = "█" * filled + "░" * (bar_width - filled)
    t = Text()
    t.append(f"  {label:12s} ", style="dim")
    t.append(bar, style=color)
    t.append(f" {_format_us(seconds)}", style=f"bold {color}")
    return t


def _step_panel(
    step_idx: int, name: str, title: str, desc: str, status: str,
    code: str = "", result: str = "", elapsed: float = 0,
    gen_time: float = 0, exec_time: float = 0, tick: int = 0,
) -> Panel:
    parts = []

    if status == "pending":
        parts.append(Text("  PENDING", style="dim"))
        parts.append(Text(f"  {desc}", style="dim italic"))
    elif status == "generating":
        frame = SPINNER_FRAMES[tick % len(SPINNER_FRAMES)]
        parts.append(Text(f"  Generating code via Claude {frame}", style="yellow bold"))
        parts.append(Text(f"  {desc}", style="dim italic"))
    elif status == "executing":
        frame = SPINNER_FRAMES[tick % len(SPINNER_FRAMES)]
        parts.append(Text(f"  Executing in Monty sandbox {frame}", style="blue bold"))
        if code:
            parts.append(Text(""))
            parts.append(Syntax(code, "python", theme="monokai", line_numbers=True, word_wrap=True))
    elif status == "done":
        parts.append(Text(f"  COMPLETE", style="green bold"))
        parts.append(Text(""))

        # Timing breakdown
        max_t = max(gen_time, exec_time, 0.001)
        parts.append(_timing_bar("Claude API", gen_time, max_t, "yellow"))
        parts.append(_timing_bar("Monty Exec", exec_time, max_t, "green"))

        if exec_time > 0 and gen_time > 0:
            speedup = gen_time / exec_time
            parts.append(Text(""))
            t = Text()
            t.append("  Sandbox executed in ", style="dim")
            t.append(f"{_format_us(exec_time)}", style="bold green")
            t.append(f"  ({speedup:,.0f}x faster than code generation)", style="dim")
            parts.append(t)

        if code:
            parts.append(Text(""))
            parts.append(Syntax(code, "python", theme="monokai", line_numbers=True, word_wrap=True))

        if result:
            parts.append(Text(""))
            parts.append(Text(result, style="bold white"))

    elif status == "error":
        parts.append(Text("  ERROR", style="red bold"))
        if result:
            parts.append(Text(""))
            parts.append(Text(result, style="red"))

    border = {
        "pending": "dim", "generating": "yellow",
        "executing": "blue", "done": "green", "error": "red",
    }.get(status, "white")

    title_str = f"[{border}]Step {step_idx}/{NUM_STEPS}: {title}[/{border}]"

    return Panel(Group(*parts), title=title_str, border_style=border, padding=(0, 1))


def _audit_panel(logs: list[tuple]) -> Panel:
    table = Table(show_header=True, header_style="bold", expand=True, show_lines=False)
    table.add_column("Time", style="dim cyan", width=10)
    table.add_column("Step", style="bold", width=14)
    table.add_column("Event", style="white")
    table.add_column("Duration", style="bold green", justify="right", width=14)

    for entry in logs[-10:]:
        table.add_row(*entry)

    if not logs:
        table.add_row("--:--:--", "--", "Waiting for pipeline to start...", "--")

    return Panel(table, title="[dim]Audit Trail[/dim]", border_style="dim")


def _build_layout(records, step_states, audit_logs, pipeline_start, tick, steps_done):
    panels = [_header_panel(pipeline_start, steps_done)]
    panels.append(_data_panel(records))
    for idx, (name, title, desc) in enumerate(STEPS, 1):
        s = step_states[name]
        panels.append(_step_panel(idx, name, title, desc, tick=tick, **s))
    panels.append(_audit_panel(audit_logs))
    return Group(*panels)


def _log(audit_logs: list, step: str, event: str, duration: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    audit_logs.append((ts, step, event, duration))


def _format_result(step_name: str, result: dict) -> str:
    """Format step result into a human-readable summary."""
    if step_name == "validate":
        return f"Valid: {result['valid_count']}  |  Invalid: {result['invalid_count']}"
    elif step_name == "normalize":
        transforms = result.get("transformations", [])
        return f"Transformations: {', '.join(transforms)}  |  Records: {len(result['records'])}"
    elif step_name == "deidentify":
        return f"Masked fields: {result['fields_masked']}  |  Records: {len(result['records'])}"
    elif step_name == "enrich":
        flags = result.get("flags_added", [])
        return f"Flags added: {', '.join(flags)}  |  Records: {len(result['records'])}"
    elif step_name == "risk_score":
        high = result.get("high_risk_count", 0)
        avg = result.get("avg_risk_score", 0)
        return f"High-risk patients: {high}  |  Avg risk score: {avg}  |  Records: {len(result['records'])}"
    elif step_name == "aggregate":
        parts = [f"Records: {result['total_records']}"]
        parts.append(f"Avg glucose: {result.get('avg_glucose', 'N/A')}")
        if "risk_distribution" in result:
            parts.append(f"Risk distribution: {result['risk_distribution']}")
        if "max_risk_score" in result:
            parts.append(f"Max risk score: {result['max_risk_score']}")
        parts.append(f"Diagnosis counts: {result['diagnosis_counts']}")
        return "  |  ".join(parts[:2]) + "\n" + "  |  ".join(parts[2:])
    return str(result)


def _get_next_records(step_name: str, result: dict, current_records: list) -> list:
    """Get the records to pass to the next pipeline step."""
    key = RECORD_PASSTHROUGH.get(step_name)
    if key and key in result:
        return result[key]
    return current_records


def _final_summary(step_states: dict, pipeline_elapsed: float):
    """Print the final performance summary after Live ends."""
    console.print()
    console.print(Rule("[bold cyan]Pipeline Complete[/bold cyan]", style="cyan"))
    console.print()

    # Performance table
    perf = Table(title="Performance Breakdown", show_header=True, header_style="bold", expand=False)
    perf.add_column("#", style="dim", width=3)
    perf.add_column("Step", style="cyan")
    perf.add_column("Code Gen (Claude)", style="yellow", justify="right")
    perf.add_column("Execution (Monty)", style="green", justify="right")
    perf.add_column("Speedup", style="bold magenta", justify="right")

    total_gen = 0
    total_exec = 0
    for idx, (name, title, _) in enumerate(STEPS, 1):
        s = step_states[name]
        gt = s.get("gen_time", 0)
        et = s.get("exec_time", 0)
        total_gen += gt
        total_exec += et
        speedup = f"{gt / et:,.0f}x" if et > 0 and gt > 0 else "N/A"
        perf.add_row(str(idx), title, _format_us(gt), _format_us(et), speedup)

    perf.add_section()
    total_speedup = f"{total_gen / total_exec:,.0f}x" if total_exec > 0 and total_gen > 0 else "N/A"
    perf.add_row(
        "", "[bold]TOTAL[/bold]",
        f"[bold]{_format_us(total_gen)}[/bold]",
        f"[bold]{_format_us(total_exec)}[/bold]",
        f"[bold]{total_speedup}[/bold]",
    )

    console.print(perf)
    console.print()

    # Security summary
    done_count = sum(1 for s in step_states.values() if s["status"] == "done")
    console.print(Panel(
        f"[bold green]All {done_count} pipeline steps executed successfully inside Monty sandbox.[/bold green]\n\n"
        "[bold]Security guarantees enforced:[/bold]\n"
        "  [red]BLOCKED[/red]  Filesystem access (open, read, write)\n"
        "  [red]BLOCKED[/red]  Network access (HTTP, sockets)\n"
        "  [red]BLOCKED[/red]  Environment variables (os.environ)\n"
        "  [red]BLOCKED[/red]  System calls (subprocess, os.system)\n"
        "  [red]BLOCKED[/red]  Dangerous builtins (exec, eval, __import__)\n"
        "  [green]ALLOWED[/green]  Pure computation (dicts, lists, functions, loops)\n\n"
        f"[dim]Total pipeline time: {pipeline_elapsed:.1f}s  |  "
        f"Code generation: {_format_us(total_gen)}  |  "
        f"Sandbox execution: {_format_us(total_exec)}[/dim]",
        title="[green]Security & Summary[/green]",
        border_style="green",
    ))


def run_pipeline():
    """Run the full healthcare data pipeline with live Rich terminal UI."""
    records = FAKE_PATIENTS
    audit_logs: list[tuple] = []
    steps_done = 0

    step_states = {}
    for name, _, _ in STEPS:
        step_states[name] = {
            "status": "pending", "code": "", "result": "", "elapsed": 0,
            "gen_time": 0, "exec_time": 0,
        }

    # Intro animation
    console.print()
    console.print(Rule("[bold cyan]Starting Healthcare AI Agent Pipeline[/bold cyan]", style="cyan"))
    console.print()
    time.sleep(0.3)

    pipeline_start = time.time()
    tick = 0

    with Live(
        _build_layout(records, step_states, audit_logs, pipeline_start, tick, steps_done),
        console=console, refresh_per_second=10, transient=False,
    ) as live:
        current_records = records

        for step_name, step_title, step_desc in STEPS:
            prompt = STEP_PROMPTS[step_name]

            # Phase 1: Generate code via Claude
            step_states[step_name]["status"] = "generating"
            _log(audit_logs, step_name, "Calling Claude for code generation...")
            tick += 1
            live.update(_build_layout(records, step_states, audit_logs, pipeline_start, tick, steps_done))

            t0 = time.time()
            try:
                code = generate_code(step_name, prompt)
            except Exception as e:
                step_states[step_name]["status"] = "error"
                step_states[step_name]["result"] = f"Code generation failed: {e}"
                _log(audit_logs, step_name, f"ERROR: {e}")
                live.update(_build_layout(records, step_states, audit_logs, pipeline_start, tick, steps_done))
                continue
            gen_time = time.time() - t0

            step_states[step_name]["code"] = code
            step_states[step_name]["gen_time"] = gen_time
            _log(audit_logs, step_name, f"Code generated ({len(code)} chars)", _format_us(gen_time))
            tick += 1
            live.update(_build_layout(records, step_states, audit_logs, pipeline_start, tick, steps_done))
            time.sleep(0.3)

            # Phase 2: Execute in Monty sandbox
            step_states[step_name]["status"] = "executing"
            _log(audit_logs, step_name, "Executing in Monty sandbox...")
            tick += 1
            live.update(_build_layout(records, step_states, audit_logs, pipeline_start, tick, steps_done))

            t1 = time.time()
            try:
                result = execute_in_sandbox(code, current_records)
            except Exception as e:
                step_states[step_name]["status"] = "error"
                step_states[step_name]["result"] = f"Sandbox execution failed: {e}"
                _log(audit_logs, step_name, f"SANDBOX ERROR: {e}")
                live.update(_build_layout(records, step_states, audit_logs, pipeline_start, tick, steps_done))
                continue
            exec_time = time.time() - t1
            total_time = time.time() - t0

            summary = _format_result(step_name, result)
            current_records = _get_next_records(step_name, result, current_records)

            steps_done += 1
            step_states[step_name]["status"] = "done"
            step_states[step_name]["result"] = summary
            step_states[step_name]["elapsed"] = total_time
            step_states[step_name]["exec_time"] = exec_time
            _log(audit_logs, step_name, f"PASS  gen={_format_us(gen_time)}  exec={_format_us(exec_time)}", _format_us(total_time))
            tick += 1
            live.update(_build_layout(records, step_states, audit_logs, pipeline_start, tick, steps_done))
            time.sleep(0.5)

        _log(audit_logs, "pipeline", "All steps complete.", _format_us(time.time() - pipeline_start))
        tick += 1
        live.update(_build_layout(records, step_states, audit_logs, pipeline_start, tick, steps_done))
        time.sleep(0.5)

    pipeline_elapsed = time.time() - pipeline_start
    _final_summary(step_states, pipeline_elapsed)


def run_pipeline_headless() -> dict:
    """Run pipeline without Rich UI — used by tests."""
    records = FAKE_PATIENTS
    results = {}

    current_records = records
    for step_name, _, _ in STEPS:
        prompt = STEP_PROMPTS[step_name]
        code = generate_code(step_name, prompt)
        result = execute_in_sandbox(code, current_records)
        results[step_name] = result
        current_records = _get_next_records(step_name, result, current_records)

    return results
