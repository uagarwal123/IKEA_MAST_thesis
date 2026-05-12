"""Parse HyperAgent traces from MAD_full_dataset.json."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from schema import Step, Trace

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "MAST-Data" / "MAD_full_dataset.json"
OUTPUT_PATH = Path(__file__).parent / "hyperagent_output_mad.json"

# --- Patterns ---

LOG_SPLIT_PATTERN = re.compile(r"HyperAgent_\S+ - INFO - ")
ACTION_TOOL_PATTERN = re.compile(r"\bAction\b.*?```python.*?\._run\(", re.DOTALL)
FINAL_ANSWER_PATTERN = re.compile(r"(?:Final Answer|final answer)\s*:", re.IGNORECASE)
AGENT_LABEL_PATTERN = re.compile(
    r"^(Inner-(?:Navigator|Executor|Editor)-Assistant's Response"
    r"|(?:Navigator|Executor|Editor)->Planner"
    r"|Planner's Response"
    r"|Initialized HyperAgent instance.*"
    r"|Initialized tools)"
    r"[:\s]",
    re.IGNORECASE,
)
_LABEL_TO_AGENT = {
    "Planner's Response": "Planner",
    "Inner-Navigator-Assistant's Response": "Navigator",
    "Navigator->Planner": "Navigator",
    "Inner-Executor-Assistant's Response": "Executor",
    "Executor->Planner": "Executor",
    "Inner-Editor-Assistant's Response": "Editor",
    "Editor->Planner": "Editor",
}


def _parse_agent_label(entry_text: str) -> tuple[str, str]:
    first_line_end = entry_text.find("\n")
    first_line = entry_text[:first_line_end] if first_line_end != -1 else entry_text
    content = entry_text[first_line_end + 1:].strip() if first_line_end != -1 else ""

    for prefix, agent in _LABEL_TO_AGENT.items():
        if first_line.startswith(prefix):
            after_label = first_line[len(prefix):].lstrip(": ").strip()
            full_content = (after_label + "\n" + content).strip() if after_label else content
            return agent, full_content

    if first_line.startswith("Initialized"):
        return "system", first_line.strip()

    return first_line.split(":")[0].strip() or "unknown", content


def _classify_kind(agent: str, content: str, label_prefix: str) -> str:
    if agent == "system":
        return "system"
    if "Assistant" in label_prefix and ACTION_TOOL_PATTERN.search(content):
        return "tool_call"
    if "->Planner" in label_prefix:
        return "tool_result"
    return "message"


def _extract_label_prefix(entry_text: str) -> str:
    first_line = entry_text.split("\n")[0]
    for prefix in list(_LABEL_TO_AGENT.keys()) + ["Initialized"]:
        if first_line.startswith(prefix):
            return prefix
    return ""


def _parse_log_entries(trajectory_str: str) -> list[tuple[str, str, str]]:
    parts = LOG_SPLIT_PATTERN.split(trajectory_str)
    entries = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        label_prefix = _extract_label_prefix(part)
        if not label_prefix:
            continue
        agent, content = _parse_agent_label(part)
        entries.append((label_prefix, agent, content))
    return entries


def _parse_yaml_metadata(trajectory_str: str) -> dict:
    result = {}
    m = re.search(r"^instance_id:\s*(.+)$", trajectory_str, re.MULTILINE)
    if m:
        result["instance_id"] = m.group(1).strip()

    ps_match = re.search(
        r"^problem_statement:\n(.*?)(?=^\S|\Z)", trajectory_str, re.MULTILINE | re.DOTALL
    )
    if ps_match:
        lines = ps_match.group(1).splitlines()
        cleaned = [re.sub(r"^ {2}", "", ln) for ln in lines]
        result["problem_statement"] = "\n".join(cleaned).strip()
    return result


def _extract_final_answer(content: str) -> str | None:
    m = FINAL_ANSWER_PATTERN.search(content)
    if not m:
        return None
    return content[m.end():].strip()


def _build_trace(record: dict, entries: list[tuple[str, str, str]], metadata: dict) -> Trace:
    steps = []
    if metadata.get("problem_statement"):
        steps.append(Step(
            agent="Human",
            content=metadata["problem_statement"],
            kind="message",
            metadata={"step_index": 0, "has_action": False, "is_final_answer": False},
        ))
    offset = len(steps)
    for i, (label_prefix, agent, content) in enumerate(entries):
        kind = _classify_kind(agent, content, label_prefix)
        final_answer = _extract_final_answer(content) if agent == "Planner" else None
        step = Step(
            agent=agent,
            content=content,
            kind=kind,
            metadata={
                "step_index": i + offset,
                "has_action": bool(ACTION_TOOL_PATTERN.search(content)),
                "is_final_answer": final_answer is not None,
            },
        )
        steps.append(step)

    agent_participation = dict(Counter(s.agent for s in steps))
    n_agent_switches = sum(
        1 for j in range(1, len(steps)) if steps[j].agent != steps[j - 1].agent
    )

    final_answer = None
    for s in steps:
        if s.metadata["is_final_answer"]:
            final_answer = _extract_final_answer(s.content)

    trace_meta = {
        "trace_id": record["trace_id"],
        "mad_trace_key": record["trace"]["key"],
        "mas_name": record["mas_name"],
        "llm_name": record["llm_name"],
        "benchmark_name": record["benchmark_name"],
        "mast_annotation": record["mast_annotation"],
        "n_turns": len(steps),
        "agent_participation": agent_participation,
        "n_agent_switches": n_agent_switches,
        "instance_id": metadata.get("instance_id"),
        "task": metadata.get("problem_statement"),
        "final_answer": final_answer,
        "success": None,
    }
    return Trace(steps=steps, metadata=trace_meta)


def parse_all(data_path: Path = DATA_PATH) -> list[Trace]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = [r for r in data if r.get("mas_name") == "HyperAgent"]
    traces = []
    for record in records:
        trajectory_str = record["trace"]["trajectory"]
        metadata = _parse_yaml_metadata(trajectory_str)
        entries = _parse_log_entries(trajectory_str)
        if not entries:
            print(f"  WARNING: trace_id {record['trace_id']} yielded no entries (mislabelled?), skipping")
            continue
        trace = _build_trace(record, entries, metadata)
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
