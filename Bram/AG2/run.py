import csv
import os
import sys
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import re
import yaml
import autogen
from openai import APIConnectionError, APIStatusError


sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from parsers.ag2_parser.ag2_parser import _build_trace, _trace_to_dict
from utils.eval import answers_match, is_unanswerable_response
from data.gsm_plus import create_sample as create_gsm_plus_sample
from data.olympiad import create_sample as create_olympiad_sample

load_dotenv()

CSV_COLUMNS = [
    "trace_id", "question_id", "model", "correct",
    "predicted_answer", "gold_answer",
    "tokens_in", "tokens_out", "latency_s", "has_code", "cost_usd",
]


def _is_termination_msg_mathchat(msg):
    content = msg.get("content", "") or ""
    has_code = bool(re.search(r"```python|'''python", content))
    if has_code:
        return False  # always execute code before terminating
    if "TERMINATE" in content:
        return True
    return bool(re.search(r"\\boxed\{[^}]+\}", content))


def _append_json_list(path: Path, item):
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    data.append(item)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_csv_row(path: Path, row: dict):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)


def run(config_path: Path):
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))

    stage = cfg["stage"]
    max_auto_reply = cfg["max_auto_reply"]
    mathchat_first_message = (config_path.parent / cfg["prompt"]).read_text(encoding="utf-8")

    llm_config = {
        "config_list": [
            {
                "model": cfg["model"]["name"],
                "api_key": os.environ["UVA_API_KEY"],
                "base_url": cfg["model"]["base_url"],
            }
        ],
        "temperature": cfg["model"]["temperature"],
    }

    benchmark = cfg.get("benchmark", "gsm_plus")
    if benchmark == "olympiad":
        data_path = create_olympiad_sample(cfg["n"])
    else:
        data_path = create_gsm_plus_sample(cfg["n"])
    with open(data_path, encoding="utf-8") as f:
        examples = json.load(f)

    model_name = llm_config["config_list"][0]["model"]
    model_slug = re.sub(r"[^a-zA-Z0-9]", "", model_name)
    date_str = datetime.now().strftime("%Y%m%d")
    run_id = f"{stage}_{benchmark}_{model_slug}_n{len(examples)}_{date_str}"

    run_dir = Path(__file__).parent / "results" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    code_base_dir = run_dir / "code"

    with open(run_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    raw_path = run_dir / "raw_traces.json"
    parsed_path = run_dir / "parsed_traces.json"
    csv_path = run_dir / "summary.csv"

    # Resume: collect already-completed question indices from existing CSV
    completed_indices = set()
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                tid = row.get("trace_id", "")
                m = re.search(r"_(\d+)$", tid)
                if m:
                    completed_indices.add(int(m.group(1)))
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()

    all_parsed = []

    for i, example in enumerate(examples):
        task_id = example.get("id", i)
        task = example["question"]
        expected_answer = example["answer"]
        perturbation_type = example.get("perturbation_type") or example.get("category", "unknown")

        code_dir = code_base_dir / f"q_{i}"

        assistant = autogen.AssistantAgent(
            name="assistant",
            llm_config=llm_config,
        )
        user_proxy = autogen.UserProxyAgent(
            name="mathproxyagent",
            is_termination_msg=_is_termination_msg_mathchat,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=max_auto_reply,
            default_auto_reply="Continue. Please keep solving the problem until you need to query. (If you get to the answer, put it in \\boxed{}.)",
            code_execution_config={"work_dir": str(code_dir), "use_docker": False, "timeout": 30},
        )

        if i in completed_indices:
            print(f"[{i+1}/{len(examples)}] task_id={task_id} -- skipped (already done)")
            continue

        print(f"[{i+1}/{len(examples)}] task_id={task_id}")
        QUESTION_TIMEOUT = 120  # seconds per question

        t_start = time.time()
        result = None
        for attempt in range(3):
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(
                user_proxy.initiate_chat,
                assistant,
                message=mathchat_first_message + task,
            )
            try:
                result = future.result(timeout=QUESTION_TIMEOUT)
                break
            except FuturesTimeoutError:
                print(f"  Question {i} timed out after {QUESTION_TIMEOUT}s, skipping.")
                break
            except (APIConnectionError, APIStatusError) as e:
                print(f"  API error (attempt {attempt+1}/3): {e}")
                if attempt < 2:
                    time.sleep(10)
                else:
                    print("  Skipping question after 3 failed attempts.")
            finally:
                executor.shutdown(wait=False)
        if result is None:
            continue
        latency = round(time.time() - t_start, 3)

        _append_json_list(raw_path, result.chat_history)

        usage = result.cost.get("usage_including_cached_inference", {})
        model_usage = next(
            (v for k, v in usage.items() if k != "total_cost" and isinstance(v, dict)),
            {},
        )

        n_code_executions = sum(
            1 for m in result.chat_history
            if m.get("name") == "mathproxyagent" and "exitcode:" in (m.get("content") or "")
        )
        last_content = (result.chat_history[-1].get("content") or "") if result.chat_history else ""
        terminated_normally = "\\boxed{" in last_content or "TERMINATE" in last_content

        trace_id = f"{run_id}_{i}"
        messages = [(m.get("content") or "", m.get("role", ""), m.get("name", "")) for m in result.chat_history]
        record = {
            "trace_id": trace_id,
            "trace": {"key": f"AG2_live_{trace_id}"},
            "mas_name": "AG2",
            "llm_name": model_name,
            "benchmark_name": {"gsm_plus": "GSM-Plus", "olympiad": "OlympiadBench"}.get(benchmark, benchmark),
            "mast_annotation": None,
        }
        trace = _build_trace(record, messages, header=None)
        if expected_answer is None or expected_answer == "None":
            correct = is_unanswerable_response(last_content)
        else:
            correct = answers_match(
                trace.metadata.get("final_answer"),
                expected_answer,
                symbolic=(benchmark == "olympiad"),
            )
        trace.metadata.update({
            "task": task,
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "temperature": llm_config["temperature"],
            "max_consecutive_auto_reply": max_auto_reply,
            "latency_seconds": latency,
            "total_tokens": model_usage.get("total_tokens"),
            "input_tokens": model_usage.get("prompt_tokens"),
            "output_tokens": model_usage.get("completion_tokens"),
            "estimated_cost_usd": usage.get("total_cost"),
            "n_code_executions": n_code_executions,
            "terminated_normally": terminated_normally,
            "expected_answer": expected_answer,
            "perturbation_type": perturbation_type,
            "correct": correct,
            "success": correct,
        })

        parsed_dict = _trace_to_dict(trace)
        _append_json_list(parsed_path, parsed_dict)
        all_parsed.append(parsed_dict)

        _append_csv_row(csv_path, {
            "trace_id": trace_id,
            "question_id": task_id,
            "model": model_name,
            "correct": correct,
            "predicted_answer": trace.metadata.get("final_answer"),
            "gold_answer": expected_answer,
            "tokens_in": model_usage.get("prompt_tokens"),
            "tokens_out": model_usage.get("completion_tokens"),
            "latency_s": latency,
            "has_code": n_code_executions > 0,
            "cost_usd": usage.get("total_cost"),
        })

        print(f"  -> correct={correct}, latency={latency}s")
        if code_dir.exists() and not any(code_dir.iterdir()):
            code_dir.rmdir()

    if code_base_dir.exists() and not any(code_base_dir.iterdir()):
        code_base_dir.rmdir()

    n_correct = sum(1 for t in all_parsed if t.get("metadata", {}).get("correct"))
    print(f"\nRun saved to {run_dir}")
    print(f"Score: {n_correct}/{len(examples)}")


if __name__ == "__main__":
    run(Path(__file__).parent / "config.yaml")
