import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import re
import autogen

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from parsers.ag2_parser.ag2_parser import _build_trace, _trace_to_dict
from utils.eval import answers_match

load_dotenv()

llm_config = {
    "config_list": [
        {
            "model": "gpt-4.1",
            "api_key": os.environ["UVA_API_KEY"],
            "base_url": "https://llmproxy.uva.nl",
        }
    ],
    "temperature": 0,
}

mathchat_first_message = """Let's use Python to solve a math problem.
Query requirements:
You should always use the 'print' function for the output and use fractions/radical
forms instead of decimals.
You can use packages like sympy to help you.
You must follow the formats below to write your code:
'''python
# your code
'''
First state the key idea to solve the problem. You may choose from three ways to
solve the problem:
Case 1: If the problem can be solved with Python code directly, please write a
program to solve it. You can enumerate all possible arrangements if needed.
Case 2: If the problem is mostly reasoning, you can solve it by yourself directly.
Case 3: If the problem cannot be handled in the above two ways, please follow this
process:
1. Solve the problem step by step (do not over-divide the steps).
2. Take out any queries that can be asked through Python (for example, any
calculations or equations that can be calculated).
3. Wait for me to give the results.
4. Continue if you think the result is correct. If the result is invalid or
unexpected, please correct your query or reasoning.
After all the queries are run and you get the answer, put the answer in \\boxed{}. 
Always put your final answer in \\boxed{} before writing TERMINATE, regardless of which case you use.
Problem: """

trace_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = Path(__file__).parent / "traces" / f"run_{trace_id}"
code_dir = run_dir / "code"
code_dir.mkdir(parents=True, exist_ok=True)

DATA = Path(__file__).parent / "data" / "gsm_plus_sample.json"
with open(DATA, encoding="utf-8") as f:
    examples = json.load(f)
example = examples[0]
task_id = example.get("id", 0)
task = example["question"]
expected_answer = example["answer"]
perturbation_type = example["perturbation_type"]
max_auto_reply = 10

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
)

def _is_termination_msg_mathchat(msg):
    content = msg.get("content", "") or ""
    if "TERMINATE" in content:
        return True
    has_code = bool(re.search(r"```python", content))
    return not has_code and bool(re.search(r"\\boxed\{[^}]+\}", content))

user_proxy = autogen.UserProxyAgent(
    name="mathproxyagent",
    is_termination_msg=_is_termination_msg_mathchat,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=max_auto_reply,
    default_auto_reply="Continue. Please keep solving the problem until you need to query. (If you get to the answer, put it in \\boxed{}.)",
    code_execution_config={"work_dir": str(code_dir), "use_docker": False},
)

t_start = time.time()
result = user_proxy.initiate_chat(
    assistant,
    message=mathchat_first_message + task,
)
latency = round(time.time() - t_start, 3)

with open(run_dir / "raw.json", "w", encoding="utf-8") as f:
    json.dump(result.chat_history, f, ensure_ascii=False, indent=2)

model_name = llm_config["config_list"][0]["model"]
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

messages = [(m.get("content") or "", m.get("role", ""), m.get("name", "")) for m in result.chat_history]
record = {
    "trace_id": trace_id,
    "trace": {"key": f"AG2_live_{trace_id}"},
    "mas_name": "AG2",
    "llm_name": model_name,
    "benchmark_name": "GSM-Plus",
    "mast_annotation": None,
}
trace = _build_trace(record, messages, header=None)
correct = answers_match(trace.metadata.get("final_answer"), expected_answer)
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

with open(run_dir / "parsed.json", "w", encoding="utf-8") as f:
    json.dump(_trace_to_dict(trace), f, ensure_ascii=False, indent=2)

print(f"Run saved to {run_dir}")
