"""Parse MetaGPT traces from MAD_full_dataset.json."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from schema import Step, Trace

ROOT = Path(__file__).parent.parent.parent
MAD_PATH = ROOT / "data" / "MAST-Data" / "MAD_full_dataset.json"
OUTPUT_PATH = Path(__file__).parent / "metagpt_output_mad.json"

# --- Patterns ---

_BLOCK_HEADER = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] "
    r"(FROM: Human TO: \{[^}]+\}|NEW MESSAGES:)"
)
_SEPARATOR = re.compile(
    r"\n-{10,}(?:\s*\n=== Communication Log Ended[^\n]*===)?\s*$",
    re.DOTALL,
)
_AGENT_BODY = re.compile(r"^\s*(\w+): \n?(.*)", re.DOTALL)
_CONTENT_FIELD = re.compile(r"\nCONTENT:\n(.*)", re.DOTALL)
_CODE_BLOCK = re.compile(r"```")
_PYTHON_CODE = re.compile(r"^(?:import |from |def |class |@)", re.MULTILINE)
_URL = re.compile(r"https?://\S+")


def _strip_separator(body: str) -> str:
    return _SEPARATOR.sub("", body).rstrip()


def _classify_kind(agent: str, content: str) -> str:
    return "message"


def parse_trajectory(trajectory: str, trace_id: str = "") -> Trace:
    parts = _BLOCK_HEADER.split(trajectory)

    steps: list[Step] = []
    task: str = ""

    n_blocks = (len(parts) - 1) // 3
    for i in range(n_blocks):
        base = 1 + i * 3
        timestamp = parts[base]
        header = parts[base + 1]
        body = _strip_separator(parts[base + 2])

        if header.startswith("FROM: Human"):
            m = _CONTENT_FIELD.search(body)
            content = m.group(1).strip() if m else body
            if not task:
                task = content
            step = Step(
                agent="Human",
                content=content,
                kind="message",
                metadata={
                    "step_index": len(steps),
                    "timestamp": timestamp,
                    "has_code": False,
                    "urls": [],
                },
            )
            steps.append(step)

        else:  # NEW MESSAGES
            m = _AGENT_BODY.match(body)
            if not m:
                continue
            agent = m.group(1)
            content = m.group(2).strip()
            kind = _classify_kind(agent, content)
            step = Step(
                agent=agent,
                content=content,
                kind=kind,
                metadata={
                    "step_index": len(steps),
                    "timestamp": timestamp,
                    "has_code": bool(_CODE_BLOCK.search(content) or _PYTHON_CODE.search(content)),
                    "urls": _URL.findall(content),
                },
            )
            steps.append(step)

    conversation_steps = [s for s in steps if s.kind != "system"]
    agent_participation = dict(Counter(s.agent for s in conversation_steps))
    n_agent_switches = sum(
        1 for a, b in zip(conversation_steps, conversation_steps[1:])
        if a.agent != b.agent
    )

    return Trace(
        steps=steps,
        metadata={
            "trace_id": trace_id,
            "task": task,
            "n_turns": len(steps),
            "n_agent_switches": n_agent_switches,
            "agent_participation": agent_participation,
            "final_answer": None,
            "success": None,
        },
    )


def parse_all(data_path: Path = MAD_PATH) -> list[Trace]:
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    records = [r for r in data if r["mas_name"] == "MetaGPT"]
    traces = []
    errors = []

    for rec in records:
        trajectory = rec["trace"]["trajectory"]
        trace_id = str(rec["trace_id"])
        try:
            trace = parse_trajectory(trajectory, trace_id=trace_id)
            trace.metadata["mad_trace_key"] = rec["trace"]["key"]
            trace.metadata["mas_name"] = rec["mas_name"]
            trace.metadata["llm_name"] = rec["llm_name"]
            trace.metadata["benchmark_name"] = rec["benchmark_name"]
            trace.metadata["mast_annotation"] = rec["mast_annotation"]
            traces.append(trace)
        except Exception as e:
            errors.append((rec["trace"]["key"], str(e)))

    if errors:
        print(f"Errors ({len(errors)}):")
        for key, err in errors:
            print(f"  {key}: {err}")

    return traces


def _trace_to_dict(trace: Trace) -> dict:
    meta = {**trace.metadata}
    meta["agent_participation"] = dict(meta.get("agent_participation", {}))
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
        "metadata": meta,
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
