"""Stage 3: MathChat with in-conversation JudgeAgent via AG2 Swarm.

Routing:
  math_agent --[has_code_block]--> judge_agent  (OnContextCondition, max 3x)
  math_agent --[is_terminating]--> judge_agent  (OnContextCondition, max 3x)
  math_agent --[code in reply]---> user_proxy   (AfterWork -> REVERT_TO_USER)
  user_proxy  --[after execution]-> math_agent  (global after_work, sets has_code_block)
  judge_agent ----------------------> math_agent  (AfterWork, resets flags)

Judge prompt is defined inline in _build_judge_system_prompt() below — edit there.
"""
import csv
import os
import sys
import json
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import yaml
from autogen.agentchat.contrib.swarm_agent import (
    SwarmAgent,
    OnContextCondition,
    ContextVariables,
    ContextExpression,
    AfterWork,
    AfterWorkOption,
    register_hand_off,
    initiate_swarm_chat,
)
from openai import APIConnectionError, APIStatusError

AG2_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = AG2_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(AG2_DIR))
sys.path.insert(0, str(REPO_ROOT / "LLM_models_interface"))

from llm_interface import _get_gcp_credentials
from parsers.ag2_parser.ag2_parser import _build_trace, _trace_to_dict
from utils.eval import answers_match, is_unanswerable_response
from data.gsm_plus import create_sample as create_gsm_plus_sample
from data.olympiad import create_sample as create_olympiad_sample

load_dotenv()


def _build_judge_system_prompt(definitions: str, examples: str = "") -> str:
    return (
        "You are a real-time process observer watching a colleague solve a math problem step by step. "
        "Your ONLY role is to detect behavioral and process failure modes as defined below. "
        "You are NOT a mathematician and must NEVER evaluate whether the mathematical reasoning or answer is correct. "
        "Specifically: NEVER write \\\\boxed{} in your response, NEVER write TERMINATE, do not suggest what the answer should be, "
        "do not point out mathematical errors, and do not provide alternative solutions.\n\n"
        "The conversation above shows the math problem and all steps taken so far. "
        "Analyse the most recent step for process failure modes. "
        "Only mark a failure mode if you can point to a specific behavioral example of it in the current conversation.\n\n"
        "Answer in exactly the format below between the @@ markers. "
        "Use the short labels as shown — do NOT repeat the question text in your answer:\n"
        "*** begin of your answer *** @@\n"
        "Summary: <one or two sentences on any process/behavioral issues; write 'None' if no issues>\n"
        "1.1: <yes or no>\n"
        "1.2: <yes or no>\n"
        "1.3: <yes or no>\n"
        "1.4: <yes or no>\n"
        "1.5: <yes or no>\n"
        "2.1: <yes or no>\n"
        "2.2: <yes or no>\n"
        "2.3: <yes or no>\n"
        "2.4: <yes or no>\n"
        "2.5: <yes or no>\n"
        "2.6: <yes or no>\n"
        "3.1: <yes or no>\n"
        "3.2: <yes or no>\n"
        "3.3: <yes or no>\n"
        "Feedback: <if ALL above are 'no' write exactly APPROVED and nothing else; "
        "if ANY is 'yes' address the solver directly in second person with ONE concrete process instruction in 2-4 sentences "
        "— do NOT suggest what the mathematical answer should be>\n"
        "@@*** end of your answer ***\n\n"
        "The failure mode numbers correspond to:\n"
        "1.1 Disobey Task Specification | 1.2 Disobey Role Specification | 1.3 Step Repetition | "
        "1.4 Loss of Conversation History | 1.5 Unaware of Termination Conditions | "
        "2.1 Conversation Reset | 2.2 Fail to Ask for Clarification | 2.3 Task Derailment | "
        "2.4 Information Withholding | 2.5 Ignored Other Agent's Input | 2.6 Action-Reasoning Mismatch | "
        "3.1 Premature Termination | 3.2 No or Incorrect Verification | 3.3 Weak Verification\n\n"
        "Definitions:\n"
        f"{definitions}\n\n"
        "Examples:\n"
        f"{examples}"
    )


CSV_COLUMNS = [
    "trace_id", "question_id", "model", "correct",
    "predicted_answer", "gold_answer",
    "tokens_in", "tokens_out", "latency_s", "has_code", "cost_usd",
    "n_code_judge", "n_terminate_judge",
]

QUESTION_TIMEOUT = 300  # seconds per question


def _append_json_list(path: Path, item):
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    data.append(item)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_csv_row(path: Path, row: dict):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)


def _has_code(text: str) -> bool:
    return bool(re.search(r"```python|'''python", text or ""))


def _is_terminating(text: str) -> bool:
    return bool(re.search(r"\\boxed\{[^}]+\}|TERMINATE", text or ""))


def _make_math_after_work(ctx: ContextVariables, math_agent, executor_agent, max_retries: int):
    """AfterWork for math_agent: routes to executor (code), self-routes for judge trigger, or terminates."""
    def _after_work(last_speaker, messages, groupchat):
        last_content = (messages[-1].get("content") or "") if messages else ""

        if ctx.get("terminate_judge_count", 0) >= max_retries:
            return AfterWorkOption.TERMINATE

        if _has_code(last_content):
            return executor_agent

        if _is_terminating(last_content):
            count = ctx.get("terminate_judge_count", 0)
            if count < max_retries:
                ctx.set("is_terminating", True)
                return math_agent  # self-route so OnContextCondition fires next
            return AfterWorkOption.TERMINATE

        return AfterWorkOption.TERMINATE

    return _after_work


def _make_executor_after_work(ctx: ContextVariables, judge_agent):
    """AfterWork for executor: flags code review context and routes to judge."""
    def _after_work(last_speaker, messages, groupchat):
        ctx.set("has_code_block", True)
        return judge_agent

    return _after_work


def _make_judge_after_work(ctx: ContextVariables, math_agent, max_judge_retries: int):
    """AfterWork for judge_agent: resets active flag, increments counter, returns to math_agent.

    If judge responds with APPROVED during a termination check, fast-paths to termination.
    Also terminates when APPROVED after a code-block review where math_agent's last message
    already contained TERMINATE/boxed (prevents a spurious extra math_agent reply).
    """
    def _after_work(last_speaker, messages, groupchat):
        last_content = (messages[-1].get("content") or "") if messages else ""
        approved = "APPROVED" in last_content

        if ctx.get("has_code_block", False):
            ctx.set("has_code_block", False)
            ctx.set("code_judge_count", ctx.get("code_judge_count", 0) + 1)
            if approved:
                # Find the most recent math_agent message and check for TERMINATE/boxed
                for m in reversed(messages[:-1]):
                    if m.get("name") == "math_agent":
                        if _is_terminating(m.get("content") or ""):
                            return AfterWorkOption.TERMINATE
                        break
        elif ctx.get("is_terminating", False):
            ctx.set("is_terminating", False)
            ctx.set("actual_terminate_judge_count", ctx.get("actual_terminate_judge_count", 0) + 1)
            if approved:
                return AfterWorkOption.TERMINATE
            ctx.set("terminate_judge_count", ctx.get("terminate_judge_count", 0) + 1)
        return math_agent

    return _after_work



def _run_question(
    math_prompt: str,
    judge_prompt: str,
    task: str,
    llm_config: dict,
    judge_llm_config: dict,
    code_dir: Path,
    max_auto_reply: int,
    max_judge_retries: int,
):
    ctx = ContextVariables(data={
        "has_code_block": False,
        "is_terminating": False,
        "code_judge_count": 0,
        "terminate_judge_count": 0,
        "actual_terminate_judge_count": 0,
    })

    math_agent = SwarmAgent(
        name="math_agent",
        llm_config=llm_config,
        code_execution_config=False,
    )
    executor_agent = SwarmAgent(
        name="code_executor",
        llm_config=False,
        human_input_mode="NEVER",
        is_termination_msg=lambda _: False,
        code_execution_config={"work_dir": str(code_dir), "use_docker": False, "timeout": 30},
        default_auto_reply="",
    )
    judge_agent = SwarmAgent(
        name="judge_agent",
        system_message=judge_prompt,
        llm_config=judge_llm_config,
        code_execution_config=False,
    )

    math_after_work = _make_math_after_work(ctx, math_agent, executor_agent, max_judge_retries)
    executor_after_work = _make_executor_after_work(ctx, judge_agent)
    judge_after_work = _make_judge_after_work(ctx, math_agent, max_judge_retries)

    register_hand_off(math_agent, hand_to=[
        OnContextCondition(
            judge_agent,
            ContextExpression(
                f"${{is_terminating}} and ${{terminate_judge_count}} < {max_judge_retries}"
            ),
        ),
        AfterWork(math_after_work),
    ])
    register_hand_off(executor_agent, hand_to=[AfterWork(executor_after_work)])
    register_hand_off(judge_agent, hand_to=[AfterWork(judge_after_work)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        chat_result, final_ctx, _ = initiate_swarm_chat(
            initial_agent=math_agent,
            messages=[{"role": "user", "content": math_prompt + task}],
            agents=[math_agent, executor_agent, judge_agent],
            user_agent=None,
            context_variables=ctx,
            after_work=AfterWorkOption.TERMINATE,
            max_rounds=60,
            exclude_transit_message=True,
        )

    return chat_result, final_ctx


def run(config_path: Path):
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))

    stage = cfg["stage"]
    max_auto_reply = cfg.get("max_auto_reply", 10)
    max_judge_retries = cfg.get("max_judge_retries", 3)

    config_dir = config_path.parent
    math_prompt = (config_dir / cfg["prompt"]).read_text(encoding="utf-8")

    definitions_path = REPO_ROOT / "data" / "prompts" / "definitions.txt"
    examples_path = REPO_ROOT / "data" / "prompts" / "examples.txt"
    definitions = definitions_path.read_text(encoding="utf-8")
    fm_examples = examples_path.read_text(encoding="utf-8") if examples_path.exists() else ""
    judge_prompt = _build_judge_system_prompt(definitions, fm_examples)

    llm_config = {
        "config_list": [{
            "model": cfg["model"]["name"],
            "api_key": os.environ["UVA_API_KEY"],
            "base_url": cfg["model"]["base_url"],
        }],
        "temperature": cfg["model"]["temperature"],
    }

    judge_cfg = cfg.get("judge_model")
    if judge_cfg:
        gcp_project = judge_cfg["project"]
        gcp_location = judge_cfg["location"]
        credentials = _get_gcp_credentials(gcp_project)
        judge_llm_config = {
            "config_list": [{
                "model": judge_cfg["name"],
                "api_type": "google",
                "project_id": gcp_project,
                "location": gcp_location,
                "credentials": credentials,
            }],
            "temperature": judge_cfg.get("temperature", 0.0),
        }
    else:
        judge_llm_config = llm_config

    benchmark = cfg.get("benchmark", "gsm_plus")
    if cfg.get("data_path"):
        data_path = Path(cfg["data_path"])
    elif benchmark == "olympiad":
        data_path = create_olympiad_sample(cfg["n"])
    else:
        data_path = create_gsm_plus_sample(cfg["n"])

    with open(data_path, encoding="utf-8") as f:
        examples = json.load(f)

    model_name = llm_config["config_list"][0]["model"]
    model_slug = re.sub(r"[^a-zA-Z0-9]", "", model_name)
    date_str = datetime.now().strftime("%Y%m%d")
    run_id = f"{stage}_{benchmark}_{model_slug}_n{len(examples)}_{date_str}"

    if cfg.get("output_dir"):
        run_dir = (config_dir / cfg["output_dir"]).resolve()
    else:
        run_dir = AG2_DIR / "results" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    code_base_dir = run_dir / "code"

    with open(run_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    raw_path = run_dir / "raw_traces.json"
    parsed_path = run_dir / "parsed_traces.json"
    csv_path = run_dir / "summary.csv"

    completed_indices = set()
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                m = re.search(r"_(\d+)$", row.get("trace_id", ""))
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

        if i in completed_indices:
            print(f"[{i+1}/{len(examples)}] task_id={task_id} -- skipped (already done)")
            continue

        print(f"[{i+1}/{len(examples)}] task_id={task_id}")

        t_start = time.time()
        chat_result = None
        final_ctx = None

        for attempt in range(3):
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(
                _run_question,
                math_prompt, judge_prompt, task,
                llm_config, judge_llm_config, code_dir, max_auto_reply, max_judge_retries,
            )
            try:
                chat_result, final_ctx = future.result(timeout=QUESTION_TIMEOUT)
                break
            except FuturesTimeoutError:
                print(f"  Q{i} timed out after {QUESTION_TIMEOUT}s, skipping.")
                break
            except (APIConnectionError, APIStatusError) as e:
                print(f"  API error (attempt {attempt+1}/3): {e}")
                if attempt < 2:
                    time.sleep(10)
                else:
                    print("  Skipping after 3 failed attempts.")
            finally:
                executor.shutdown(wait=False)

        if chat_result is None:
            continue

        latency = round(time.time() - t_start, 3)

        _append_json_list(raw_path, chat_result.chat_history)

        usage = chat_result.cost.get("usage_including_cached_inference", {})
        model_usage = next(
            (v for k, v in usage.items() if k != "total_cost" and isinstance(v, dict)),
            {},
        )

        n_code_executions = sum(
            1 for m in chat_result.chat_history
            if m.get("name") == "code_executor" and "exitcode:" in (m.get("content") or "")
        )
        last_msg = chat_result.chat_history[-1] if chat_result.chat_history else {}
        last_content = last_msg.get("content") or ""
        last_agent = last_msg.get("name", "")
        terminated_normally = (
            "\\boxed{" in last_content
            or "TERMINATE" in last_content
            or (last_agent == "judge_agent" and "APPROVED" in last_content)
        )

        n_code_judge = final_ctx.get("code_judge_count", 0) if final_ctx else 0
        n_terminate_judge = final_ctx.get("actual_terminate_judge_count", 0) if final_ctx else 0

        trace_id = f"{run_id}_{i}"
        messages = [
            (m.get("content") or "", m.get("role", ""), m.get("name") or "user")
            for m in chat_result.chat_history
        ]
        record = {
            "trace_id": trace_id,
            "trace": {"key": f"AG2_stage3_{trace_id}"},
            "mas_name": "AG2_stage3",
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
            "max_auto_reply": max_auto_reply,
            "max_judge_retries": max_judge_retries,
            "latency_seconds": latency,
            "total_tokens": model_usage.get("total_tokens"),
            "input_tokens": model_usage.get("prompt_tokens"),
            "output_tokens": model_usage.get("completion_tokens"),
            "estimated_cost_usd": usage.get("total_cost"),
            "n_code_executions": n_code_executions,
            "n_code_judge_interventions": n_code_judge,
            "n_terminate_judge_interventions": n_terminate_judge,
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
            "n_code_judge": n_code_judge,
            "n_terminate_judge": n_terminate_judge,
        })

        print(f"  -> correct={correct}, latency={latency}s, "
              f"code_judge={n_code_judge}, term_judge={n_terminate_judge}")

        if code_dir.exists() and not any(code_dir.iterdir()):
            code_dir.rmdir()

    if code_base_dir.exists() and not any(code_base_dir.iterdir()):
        code_base_dir.rmdir()

    n_correct = sum(1 for t in all_parsed if t.get("metadata", {}).get("correct"))
    print(f"\nRun saved to {run_dir}")
    print(f"Score: {n_correct}/{len(examples)}")


if __name__ == "__main__":
    run(Path(__file__).parent / "config.yaml")
