"""Parse AG2 traces from MAD_full_dataset.json."""
import ast
import json
import re
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from schema import Step, Trace

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "MAST-Data" / "MAD_full_dataset.json"
OUTPUT_PATH = Path(__file__).parent / "ag2_output_mad.json"

URL_PATTERN = re.compile(r"https?://\S+")
CODE_BLOCK_PATTERN = re.compile(r"```python", re.IGNORECASE)
BOXED_PATTERN = re.compile(r"\\boxed\{([^}]+)\}")
PROBLEM_PATTERN = re.compile(r"Problem:\n(.+)", re.DOTALL)

YAML_MSG_PATTERN = re.compile(
    r"      content:(.*?)\n      role:\s*(\S+)\s*\n      name:\s*(\S+)",
    re.DOTALL,
)


def _extract_urls(text: str) -> list[str]:
    return URL_PATTERN.findall(text)


def _classify_kind(agent: str, content: str, prev_kind: str | None) -> str:
    if agent == "chat_manager":
        return "system"
    has_code = bool(CODE_BLOCK_PATTERN.search(content))
    if has_code:
        return "tool_call"
    if prev_kind == "tool_call":
        return "tool_result"
    return "message"


def _extract_final_answer(content: str) -> str | None:
    m = BOXED_PATTERN.search(content)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return None


def _make_step(agent: str, role: str, content: str, step_index: int, prev_kind: str | None) -> Step:
    if step_index == 0 and agent == "mathproxyagent":
        kind = "system"
    else:
        kind = _classify_kind(agent, content, prev_kind)
    final_answer = _extract_final_answer(content) if kind != "system" else None
    return Step(
        agent=agent,
        content=content,
        kind=kind,
        metadata={
            "step_index": step_index,
            "role": role,
            "has_code": bool(CODE_BLOCK_PATTERN.search(content)),
            "is_final_answer": final_answer is not None,
            "urls": _extract_urls(content),
        },
    )


def _parse_yaml_header(header_str: str) -> dict:
    result = {}
    for key in ["instance_id", "problem_statement"]:
        m = re.search(rf"^{key}:\s*(.+)$", header_str, re.MULTILINE)
        result[key] = m.group(1).strip() if m else None
    for key in ["correct", "answer", "given", "perturbation_type", "seed_answer"]:
        m = re.search(rf"^  {key}:\s*(.+)$", header_str, re.MULTILINE)
        if m:
            val = m.group(1).strip()
            if val == "True":
                val = True
            elif val == "False":
                val = False
            elif val == "None":
                val = None
            result[f"other_data.{key}"] = val
        else:
            result[f"other_data.{key}"] = None
    return result


def _clean_yaml_content(raw: str) -> str:
    lines = raw.splitlines()
    cleaned = []
    for line in lines:
        if line.startswith(" " * 12):
            cleaned.append(line[12:])
        else:
            cleaned.append(line.lstrip())
    return "\n".join(cleaned).strip()


def _parse_yaml_trajectory(trajectory_str: str) -> tuple[dict, list[tuple[str, str, str]]]:
    """Returns (header_dict, [(content, role, name), ...])."""
    parts = trajectory_str.split("\ntrajectory:\n", 1)
    header = _parse_yaml_header(parts[0])
    body = parts[1] if len(parts) > 1 else ""
    messages = []
    for raw_content, role, name in YAML_MSG_PATTERN.findall(body):
        content = _clean_yaml_content(raw_content)
        messages.append((content, role, name))
    return header, messages


def _parse_dictrepr_trajectory(trajectory_str: str) -> list[tuple[str, str, str]]:
    """Returns [(content, role, name), ...]."""
    chunks = re.split(r"\} \{'content':", trajectory_str)
    messages = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            chunk = "{'content':" + chunk
        if i < len(chunks) - 1:
            chunk = chunk + "}"
        try:
            msg = ast.literal_eval(chunk)
        except Exception:
            continue
        content_raw = msg.get("content", "")
        if isinstance(content_raw, list):
            content = "\n".join(str(c) for c in content_raw)
        else:
            content = str(content_raw)
        role = msg.get("role", "")
        name = msg.get("name", "")
        messages.append((content, role, name))
    return messages


def _build_trace(record: dict, messages: list[tuple[str, str, str]], header: dict | None) -> Trace:
    steps = []
    prev_kind = None
    for i, (content, role, name) in enumerate(messages):
        step = _make_step(name, role, content, i, prev_kind)
        steps.append(step)
        prev_kind = step.kind

    agent_participation = dict(Counter(s.agent for s in steps))
    n_turns = len(steps)
    n_agent_switches = sum(
        1 for j in range(1, len(steps)) if steps[j].agent != steps[j - 1].agent
    )
    trace_meta: dict = {
        "trace_id": record["trace_id"],
        "mad_trace_key": record["trace"]["key"],
        "mas_name": record["mas_name"],
        "llm_name": record["llm_name"],
        "benchmark_name": record["benchmark_name"],
        "mast_annotation": record["mast_annotation"],
        "n_turns": n_turns,
        "agent_participation": agent_participation,
        "n_agent_switches": n_agent_switches,
        "task": None,
        "final_answer": None,
        "success": None,
    }

    if header:
        trace_meta["instance_id"] = header.get("instance_id")
        trace_meta["task"] = header.get("problem_statement")
        trace_meta["success"] = header.get("other_data.correct")
        trace_meta["answer"] = header.get("other_data.answer")
        trace_meta["final_answer"] = header.get("other_data.given")
        trace_meta["perturbation_type"] = header.get("other_data.perturbation_type")
        trace_meta["seed_answer"] = header.get("other_data.seed_answer")
    else:
        if steps:
            m = PROBLEM_PATTERN.search(steps[0].content)
            if m:
                trace_meta["task"] = m.group(1).strip()
            elif steps[0].kind == "message":
                trace_meta["task"] = steps[0].content.strip()
        for step in reversed(steps):
            fa = _extract_final_answer(step.content)
            if fa:
                trace_meta["final_answer"] = fa
                break

    return Trace(steps=steps, metadata=trace_meta)


def parse_all(data_path: Path = DATA_PATH) -> list[Trace]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ag2_records = [r for r in data if r.get("mas_name") == "AG2"]
    traces = []

    for record in ag2_records:
        trajectory_str = record["trace"]["trajectory"]
        is_yaml = "\ntrajectory:\n" in trajectory_str

        if is_yaml:
            header, messages = _parse_yaml_trajectory(trajectory_str)
        else:
            header = None
            messages = _parse_dictrepr_trajectory(trajectory_str)

        if not messages:
            continue

        trace = _build_trace(record, messages, header)
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
