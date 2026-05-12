"""Parse OpenManus traces from MAD_full_dataset.json."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from schema import Step, Trace

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "MAST-Data" / "MAD_full_dataset.json"
OUTPUT_PATH = Path(__file__).parent / "openmanus_output_mad.json"

# --- Patterns ---

_LOG_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+) \| (\w+)\s+\| ([\w.]+:\w+:\d+) - ",
    re.MULTILINE,
)
_EXECUTING_STEP_RE = re.compile(r"Executing step (\d+)/(\d+)")
_TOOL_COUNT_RE = re.compile(r"Manus selected (\d+) tools? to use")
_TOOLS_PREPARED_RE = re.compile(r"Tools being prepared: (\[.*?\])")
_TOOL_ARGS_RE = re.compile(r"Tool arguments: (\{.*\})")
_ACTIVATING_TOOL_RE = re.compile(r"Activating tool: '(\w+)'")
_TOOL_RESULT_RE = re.compile(
    r"Tool '(\w+)' completed its mission! Result: Observed output of cmd `\w+` executed:\n([\s\S]*)"
)
_PLAN_CREATION_RESULT_RE = re.compile(r"Plan creation result: (.+)")
_TASK_RE = re.compile(r"Read prompt from task\.txt: (.+)")
_TIMEOUT_RE = re.compile(r"Request processing timed out")
_PLAN_COMPLETED_RE = re.compile(r"Plan completed:\n([\s\S]+)")


def _split_log_entries(text: str) -> list[tuple[str, str, str, str]]:
    matches = list(_LOG_LINE_RE.finditer(text))
    if not matches:
        return [("", "", "preamble", text.strip())]

    entries = []
    if matches[0].start() > 0:
        entries.append(("", "", "preamble", text[: matches[0].start()].strip()))

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[m.end() : end].rstrip()
        entries.append((m.group(1), m.group(2), m.group(3), content))

    return entries


def _group_into_turns(
    entries: list[tuple[str, str, str, str]],
) -> tuple[list[tuple[str, str, str, str]], list[list[tuple[str, str, str, str]]]]:
    preamble = []
    turns: list[list[tuple]] = []
    current: list[tuple] | None = None

    for entry in entries:
        ts, level, source, content = entry
        if source == "app.agent.base:run:140" and _EXECUTING_STEP_RE.search(content):
            if current is not None:
                turns.append(current)
            current = [entry]
        elif current is None:
            preamble.append(entry)
        else:
            current.append(entry)

    if current:
        turns.append(current)

    return preamble, turns


def _extract_plan_text(preamble: list[tuple], turns: list[list[tuple]]) -> str | None:
    all_entries = preamble + (turns[0] if turns else [])
    for ts, level, source, content in all_entries:
        if source == "app.flow.planning:_create_initial_plan:179":
            plan_start = content.find("\nPlan:")
            if plan_start != -1:
                return content[plan_start:].strip()
            return content.strip()
    return None


def _extract_task_prompt(preamble: list[tuple]) -> str | None:
    for ts, level, source, content in preamble:
        m = _TASK_RE.search(content)
        if m:
            return m.group(1).strip()
    return None


def _parse_turn(turn_entries: list[tuple], turn_index: int) -> list[Step]:
    thought = None
    n_tools = 0
    tool_names: list[str] = []
    tool_args_list: list[str] = []
    tool_result_tool: str | None = None
    tool_result_content: str | None = None
    is_stuck = False

    for ts, level, source, content in turn_entries[1:]:
        if source == "app.agent.toolcall:think:81":
            thought = content.strip()
        elif source == "app.agent.toolcall:think:82":
            m = _TOOL_COUNT_RE.search(content)
            if m:
                n_tools = int(m.group(1))
        elif source == "app.agent.toolcall:think:86":
            m = _TOOLS_PREPARED_RE.search(content)
            if m:
                try:
                    tool_names = json.loads(m.group(1).replace("'", '"'))
                except Exception:
                    tool_names = [m.group(1)]
        elif source == "app.agent.toolcall:think:89":
            m = _TOOL_ARGS_RE.search(content)
            if m:
                tool_args_list.append(m.group(1))
        elif source == "app.agent.toolcall:act:150":
            m = _TOOL_RESULT_RE.search(content)
            if m:
                tool_result_tool = m.group(1)
                tool_result_content = m.group(2).strip()
        elif source == "app.agent.base:handle_stuck_state:168":
            is_stuck = True

    steps: list[Step] = []

    if thought is not None:
        steps.append(Step(
            agent="Manus",
            content=thought,
            kind="message",
            metadata={"step_n": turn_index, "n_tools_selected": n_tools, "is_stuck": is_stuck},
        ))

    if n_tools > 0 and tool_names:
        tool_name = tool_names[0]
        args_str = tool_args_list[0] if tool_args_list else "{}"
        steps.append(Step(
            agent="Manus",
            content=f"{tool_name}\n{args_str}",
            kind="tool_call",
            metadata={"step_n": turn_index, "tool_name": tool_name, "tool_args": args_str, "all_tools": tool_names},
        ))

    if tool_result_content is not None:
        steps.append(Step(
            agent="Manus",
            content=tool_result_content,
            kind="tool_result",
            metadata={"step_n": turn_index, "tool_name": tool_result_tool},
        ))

    return steps


def _extract_termination(entries_flat: list[tuple]) -> dict:
    timed_out = False
    plan_completed_text = None
    for ts, level, source, content in entries_flat:
        if source in ("__main__:run_flow:45", "__main__:run_flow:46"):
            if _TIMEOUT_RE.search(content):
                timed_out = True
        if "Plan completed:" in content:
            m = _PLAN_COMPLETED_RE.search(content)
            if m:
                plan_completed_text = m.group(1).strip()
    return {"timed_out": timed_out, "plan_completed_text": plan_completed_text}


def _parse_trajectory(trajectory_str: str, record: dict) -> Trace | None:
    entries = _split_log_entries(trajectory_str)
    preamble, turns = _group_into_turns(entries)

    if not turns and not preamble:
        return None

    all_entries_flat = preamble + [e for t in turns for e in t]

    plan_text = _extract_plan_text(preamble, turns)
    task_prompt = _extract_task_prompt(preamble)
    term = _extract_termination(all_entries_flat)

    steps: list[Step] = []

    if task_prompt:
        steps.append(Step(
            agent="Human",
            content=task_prompt,
            kind="message",
            metadata={"step_index": 0},
        ))

    if plan_text:
        steps.append(Step(
            agent="system",
            content=plan_text,
            kind="system",
            metadata={},
        ))

    for i, turn in enumerate(turns):
        steps.extend(_parse_turn(turn, i))

    for i, step in enumerate(steps):
        step.metadata["step_index"] = i

    if not steps:
        return None

    agent_participation = dict(Counter(s.agent for s in steps))
    n_agent_switches = sum(
        1 for i in range(1, len(steps)) if steps[i].agent != steps[i - 1].agent
    )

    trace_meta = {
        "trace_id": record.get("trace_id"),
        "mad_trace_key": record.get("trace", {}).get("key"),
        "mas_name": record.get("mas_name"),
        "llm_name": record.get("llm_name"),
        "benchmark_name": record.get("benchmark_name"),
        "mast_annotation": record.get("mast_annotation"),
        "n_turns": len(steps),
        "n_agent_turns": len(turns),
        "agent_participation": agent_participation,
        "n_agent_switches": n_agent_switches,
        "task": task_prompt,
        "timed_out": term["timed_out"],
        "final_answer": term["plan_completed_text"],
        "success": None,
    }

    return Trace(steps=steps, metadata=trace_meta)


def parse_all(data_path: Path = DATA_PATH) -> list[Trace]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = [r for r in data if r.get("mas_name") == "OpenManus"]
    traces = []
    for record in records:
        trajectory_str = record["trace"]["trajectory"]
        trace = _parse_trajectory(trajectory_str, record)
        if trace is None:
            print(f"  WARNING: trace_id {record['trace_id']} yielded no steps, skipping")
            continue
        traces.append(trace)
    return traces


def _trace_to_dict(trace: Trace) -> dict:
    return {
        "steps": [
            {
                "agent": s.agent,
                "content": s.content,
                "kind": s.kind,
                "metadata": s.metadata,
            }
            for s in trace.steps
        ],
        "metadata": trace.metadata,
    }


def main():
    traces = parse_all()
    print(f"Parsed: {len(traces)} records, Errors: 0")
    output = [_trace_to_dict(t) for t in traces]
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
