"""Parse ChatDev traces from MAD_full_dataset.json."""
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
OUTPUT_PATH = Path(__file__).parent / "chatdev_output_mad.json"

# --- Patterns ---

_LOG_HEADER = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) INFO\] (.+)$",
    re.MULTILINE,
)
_TURN_HEADER = re.compile(r"\*\*(.+?)<->(.+?) on : (.+?), turn (\d+)\*\*")
_SYSTEM_MARKERS = frozenset([
    "[Preprocessing]", "[chatting]", "[RolePlaying]",
    "[Seminar Conclusion]", "[Update Codes]", "[Software Info]",
])
_TOOL_RESULT_MARKERS = frozenset([
    "[Test Reports]",
])
_FINAL_ANSWER = re.compile(r"^<INFO>\s+\w", re.MULTILINE)
_URL = re.compile(r"https?://\S+")
_COST = re.compile(r"cost:\s*\$([0-9.]+)")
_TASK_PROMPT = re.compile(r"\*\*task_prompt\*\*[:\s|]+([^\n|]+)")
_PREAMBLE = re.compile(r"^\[ChatDev is a software company.*?\]\n\n", re.DOTALL)
_DROP_MARKERS = frozenset([
    "[OpenAI_Usage_Info Receive]",
    "[Execute Detail]",
    "[Seminar Conclusion]",
])


def _classify_kind(speaker: str, header_rest: str, content: str) -> str:
    if any(m in header_rest for m in _TOOL_RESULT_MARKERS):
        return "tool_result"
    if speaker == "System" or any(m in header_rest for m in _SYSTEM_MARKERS):
        return "system"
    return "message"


def parse_log_text(text: str, trace_id: str = "") -> Trace:
    matches = list(_LOG_HEADER.finditer(text))
    entries: list[tuple[str, str, str]] = []
    for i, m in enumerate(matches):
        timestamp = m.group(1)
        rest = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        entries.append((timestamp, rest, body))

    steps: list[Step] = []
    task: str = ""
    total_cost: float = 0.0
    phases_seen: list[str] = []
    phases_order: list[str] = []

    for timestamp, rest, body in entries:
        if rest.startswith("flask app.py") or rest.startswith("HTTP Request"):
            continue

        if ": " in rest:
            speaker, header_text = rest.split(": ", 1)
        else:
            speaker = "System"
            header_text = rest

        speaker = speaker.strip()
        header_text = header_text.strip()

        if any(m in header_text for m in _DROP_MARKERS):
            continue
        if body.startswith("execute SimplePhase:") or body.startswith("execute ComposedPhase:"):
            continue

        content = _PREAMBLE.sub("", body).strip()
        kind = _classify_kind(speaker, header_text, content)

        if not task and "[Preprocessing]" in header_text:
            m = _TASK_PROMPT.search(content)
            if m:
                task = m.group(1).strip()

        cost_m = _COST.search(content)
        if cost_m:
            total_cost += float(cost_m.group(1))

        turn_m = _TURN_HEADER.search(header_text)
        phase_name = turn_m.group(3).strip() if turn_m else ""
        turn_number = int(turn_m.group(4)) if turn_m else None
        agent_pair = f"{turn_m.group(1).strip()}<->{turn_m.group(2).strip()}" if turn_m else ""

        if phase_name and phase_name not in phases_seen:
            phases_seen.append(phase_name)
            phases_order.append(phase_name)

        urls = _URL.findall(content)

        step = Step(
            agent=speaker,
            content=content,
            kind=kind,
            metadata={
                "step_index": len(steps),
                "timestamp": timestamp,
                "phase_name": phase_name,
                "turn_number": turn_number,
                "agent_pair": agent_pair,
                "has_code": kind == "message" and "```" in content,
                "is_final_answer": bool(_FINAL_ANSWER.search(content)),
                "urls": list(dict.fromkeys(urls)),
            },
        )
        steps.append(step)

    conversation_steps = [s for s in steps if s.kind == "message"]
    agent_participation: Counter = Counter(s.agent for s in conversation_steps)
    n_agent_switches = sum(
        1 for a, b in zip(conversation_steps, conversation_steps[1:])
        if a.agent != b.agent
    )

    return Trace(
        steps=steps,
        metadata={
            "trace_id": trace_id,
            "task": task,
            "phases": phases_order,
            "n_turns": len(steps),
            "n_agent_switches": n_agent_switches,
            "agent_participation": agent_participation,
            "total_cost_usd": round(total_cost, 6),
            "final_answer": None,
            "success": None,
        },
    )


def trace_to_dict(trace) -> dict:
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


def parse_all(data_path: Path = MAD_PATH) -> list[Trace]:
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    traces = []
    for rec in [r for r in data if r["mas_name"] == "ChatDev"]:
        try:
            trace = parse_log_text(rec["trace"]["trajectory"], trace_id=str(rec["trace_id"]))
            trace.metadata["mad_trace_key"] = rec["trace"]["key"]
            trace.metadata["mas_name"] = rec["mas_name"]
            trace.metadata["llm_name"] = rec["llm_name"]
            trace.metadata["benchmark_name"] = rec["benchmark_name"]
            trace.metadata["mast_annotation"] = rec["mast_annotation"]
            traces.append(trace)
        except Exception:
            pass
    return traces


def main():
    traces = parse_all()
    print(f"Parsed: {len(traces)} records, Errors: 0")
    output = [trace_to_dict(t) for t in traces]
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
