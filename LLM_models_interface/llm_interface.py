"""
LLM interface for the MAST failure-mode judge.

Wraps Anthropic (via Vertex AI), Gemini (via Vertex AI), and Ollama behind a single call: 
llm_interface.judge(trace, config) -> JudgeResponse.
"""

import time
import datetime
from pathlib import Path
from collections import defaultdict
import re
import yaml
import json
import random
from dataclasses import dataclass, field
from google import genai
from google.genai import types as genai_types
import google.auth.impersonated_credentials
import google.auth.transport.requests
from google.cloud import secretmanager
from anthropic import AnthropicVertex
import ollama
from openai import OpenAI
import os

# Model prices per 1M tokens: (price_in, price_out)
PRICES: dict[str, tuple[float, float]] = {
    "o1":                        (15.00, 60.00),
    "claude-sonnet-4-6":         ( 3.00, 15.00),
    "claude-haiku-4-5":          ( 1.00,  1.25),
    "gemini-2.5-pro":            ( 2.50, 10.00),
    "gpt-4.1":                   ( 2.00,  8.00),
    "o3-mini":                   ( 1.10,  4.40),
    "gpt-5":                     ( 1.25, 10.00),
    "grok-4.3":                  ( 1.25,  2.50),
}

FAILURE_MODES = ["1.1","1.2","1.3","1.4","1.5","2.1","2.2","2.3","2.4","2.5","2.6","3.1","3.2","3.3"]

@dataclass
class JudgeConfig:
    name: str = ""
    model: str = ""
    backend: str = "genai"   # "genai" | "anthropic" | "ollama"
    temperature: float = 0.0
    reasoning: bool = False
    shots: int = 0
    slice_n: int | None = None
    system_prompt: str = ""
    definitions_path: str = "../../data/prompts/definitions.txt"
    examples_path: str  = "../../data/prompts/examples.txt"
    dataset_path: str   = ""
    genai_project: str  = "ingka-map-services-dev"
    genai_location: str = "europe-west1"
    ollama_host: str    = "http://localhost:11434"
    uva_base_url: str   = "https://llmproxy.uva.nl"

def load_configs(path: str) -> list[JudgeConfig]:                                                                                                        
      config_path = Path(path).resolve()                                                                                                                   
      config_dir = config_path.parent                                                                                                                      
      with open(config_path) as f:                                                                                                                         
          data = yaml.safe_load(f)                                                                 
      configs = []
      for item in data["experiments"]:
          cfg = JudgeConfig(**item)
          for field in ("definitions_path", "examples_path", "dataset_path"):
              val = getattr(cfg, field)
              if val and not Path(val).is_absolute():
                  setattr(cfg, field, str(config_dir / val))
          configs.append(cfg)
      return configs

@dataclass
class JudgeResponse:
    """Holds the raw and parsed output of a single LLM call, plus usage stats."""

    trace_id: str
    raw_text: str
    model_id: str
    tokens_in: int
    tokens_out: int
    latency_s: float
    annotations: dict[str, int] = field(default_factory=dict)
    cost_usd: float = field(init=False)

    def __post_init__(self):
        price_in, price_out = PRICES.get(self.model_id, (0.0, 0.0))
        self.cost_usd = (
            self.tokens_in * price_in
            + self.tokens_out * price_out
        ) / 1_000_000


def build_judge_prompt(trace: str, definitions: str, examples: str=''):
    prompt = (
    "Below I will provide a multiagent system trace. provide me an analysis of the failure modes and inefficiencies as I will say below. \n"
    "In the traces, analyze the system behaviour."
    "There are several failure modes in multiagent systems I identified. I will provide them below. Tell me if you encounter any of them, as a binary yes or no. \n"
    "Also, give me a one sentence (be brief) summary of the problems with the inefficiencies or failure modes in the trace. Only mark a failure mode if you can provide an example of it in the trace, and specify that in your summary at the end"
    "Also tell me whether the task is successfully completed or not, as a binary yes or no."
    "At the very end, I provide you with the definitions of the failure modes and inefficiencies. After the definitions, I will provide you with examples of the failure modes and inefficiencies for you to understand them better."
    "Tell me if you encounter any of them between the @@ symbols as I will say below, as a binary yes or no."
    "Here are the things you should answer. Start after the @@ sign and end before the next @@ sign (do not include the @@ symbols in your answer):"
    "*** begin of things you should answer *** @@"
    "A. Freeform text summary of the problems with the inefficiencies or failure modes in the trace: <summary>"
    "B. Whether the task is successfully completed or not: <yes or no>"
    "C. Whether you encounter any of the failure modes or inefficiencies:"
    "1.1 Disobey Task Specification: <yes or no>"
    "1.2 Disobey Role Specification: <yes or no>"
    "1.3 Step Repetition: <yes or no>"
    "1.4 Loss of Conversation History: <yes or no>"
    "1.5 Unaware of Termination Conditions: <yes or no>"
    "2.1 Conversation Reset: <yes or no>"
    "2.2 Fail to Ask for Clarification: <yes or no>"
    "2.3 Task Derailment: <yes or no>"
    "2.4 Information Withholding: <yes or no>"
    "2.5 Ignored Other Agent's Input: <yes or no>"
    "2.6 Action-Reasoning Mismatch: <yes or no>"
    "3.1 Premature Termination: <yes or no>"
    "3.2 No or Incorrect Verification: <yes or no>"
    "3.3 Weak Verification: <yes or no>"
    "@@*** end of your answer ***"
    "An example answer is: \n"
    "A. The task is not completed due to disobeying role specification as agents went rogue and started to chat with each other instead of completing the task. Agents derailed and verifier is not strong enough to detect it.\n"
    "B. no \n"
    "C. \n"
    "1.1 no \n"
    "1.2 no \n"
    "1.3 no \n"
    "1.4 no \n"
    "1.5 no \n"
    "1.6 yes \n"
    "2.1 no \n"
    "2.2 no \n"
    "2.3 yes \n"
    "2.4 no \n"
    "2.5 no \n"
    "2.6 yes \n"
    "2.7 no \n"
    "3.1 no \n"
    "3.2 yes \n"
    "3.3 no \n"   
    "Here is the trace: \n"
    f"{trace}"
    "Also, here are the explanations (definitions) of the failure modes and inefficiencies: \n"
    f"{definitions} \n"
    "Here are some examples of the failure modes and inefficiencies: \n"
    f"{examples}"
)
    return prompt


class GCPAuth:
    def __init__(self, project_id: str, impersonate_service_account: str = None, lifetime: int = 3600):
        """
        Initializes the GCPAuth class.
        """
        self.project_id = project_id
        self.impersonate_service_account = impersonate_service_account
        self.lifetime = lifetime
        self.creds = None

    def get_remaining_lifetime(self):
        """        
        Returns the remaining lifetime of the credentials in seconds.
        If the credentials are None, returns 0.
        If the credentials are expired, returns 0.
        If the credentials are valid, returns the remaining lifetime in seconds.
        """
        if self.creds is not None:
            now = datetime.datetime.now(datetime.timezone.utc).timestamp()
            expiration = self.creds.expiry.replace(
                tzinfo=datetime.timezone.utc).timestamp()
            return max(int(expiration) - int(now), 0)
        return 0

    def is_token_expired(self):
        """       
        Checks if the token is expired.
        Returns True if the token is expired, False otherwise.
        """
        remaining_lifetime = self.get_remaining_lifetime()
        return remaining_lifetime == 0

    @staticmethod
    def access_secret_version(creds, project_id, secret_id, version_id="latest"):
        client = secretmanager.SecretManagerServiceClient(
            credentials=creds)
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode('UTF-8')

    def get_credentials(self):
        """
        Retrieves the GCP credentials.
        If the credentials are already cached and not expired, returns them.
        If the credentials are not cached or expired, authenticates and returns new credentials.
        """
        if self.creds is not None and not self.is_token_expired():
            # logger.info("Using cached credentials")
            print("Using cached credentials")
            return self.creds

        if self.creds is None:
            # logger.info("No credentials found, authenticating...")
            print("No credentials found, authenticating...")
            target_scopes = [
                "https://www.googleapis.com/auth/cloud-platform"
            ]
            creds, pid = google.auth.default(scopes=target_scopes)
            if self.impersonate_service_account is not None:
                self.creds = google.auth.impersonated_credentials.Credentials(
                    source_credentials=creds,
                    target_principal=self.impersonate_service_account,
                    target_scopes=target_scopes,
                    lifetime=self.lifetime,  # seconds up to 3600 (1h)
                )
                # logger.info(f"Authenticated with SA {self.impersonate_service_account}")
                print(
                    f"Authenticated with SA {self.impersonate_service_account}")
            else:
                self.creds = creds
                if hasattr(self.creds, "service_account_email"):
                    print(self.creds.service_account_email)
                else:
                    print("Authenticated with default credentials ")
        # logger.info("Refreshing credentials...")
        print("Refreshing credentials...")
        self.creds.refresh(google.auth.transport.requests.Request())
        return self.creds
    
_gcp_auth: GCPAuth | None = None

def _get_gcp_credentials(project: str) -> object:
    global _gcp_auth
    if _gcp_auth is None:
        _gcp_auth = GCPAuth(project_id=project)
    return _gcp_auth.get_credentials()

def _call_genai(model: str, prompt: str, temperature: float, trace_id: str, project: str, location: str, system_prompt="", reasoning: bool = False) -> JudgeResponse:
    """Call the Google GenAI API."""

    client = genai.Client(
    vertexai=True,
    project=project,
    location=location,
    credentials=_get_gcp_credentials(project),
    )

    config = genai_types.GenerateContentConfig(
        system_instruction=system_prompt or None,
        temperature=temperature,
        thinking_config=genai_types.ThinkingConfig(thinking_budget=-1) if reasoning else None,
    )

    t0 = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config
    )

    latency = time.perf_counter() - t0
    raw = response.text
    usage = response.usage_metadata
    return JudgeResponse(
        trace_id=trace_id,
        raw_text=raw,
        model_id=model,
        tokens_in=usage.prompt_token_count or 0,
        tokens_out=usage.candidates_token_count or 0,
        latency_s=latency
    )

def _call_anthropic_vertex(model: str, prompt: str, temperature: float, trace_id: str, project: str, location: str, system_prompt: str = "", reasoning: bool = False, max_tokens: int = 16000) -> JudgeResponse:
    """Call a Claude model hosted on Vertex AI Model Garden."""

    client = AnthropicVertex(
        region=location,
        project_id=project,
        credentials=_get_gcp_credentials(project),
    )

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if reasoning:
        # Extended thinking requires budget_tokens and disables temperature
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}
    else:
        kwargs["temperature"] = temperature
    if system_prompt:
        kwargs["system"] = system_prompt

    t0 = time.perf_counter()
    response = client.messages.create(**kwargs)
    latency = time.perf_counter() - t0

    raw = next((b.text for b in response.content if b.type == "text"), "")
    return JudgeResponse(
        trace_id=trace_id,
        raw_text=raw,
        model_id=model,
        tokens_in=response.usage.input_tokens or 0,
        tokens_out=response.usage.output_tokens or 0,
        latency_s=latency,
    )


def _call_ollama(model: str, prompt: str, temperature: float, trace_id: str, host: str, system_prompt="") -> JudgeResponse:
    """Call a local Ollama model."""

    client = ollama.Client(host=host)
    t0 = time.perf_counter()
    messages=[]
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature}
    )
    latency = time.perf_counter() - t0
    raw = response.message.content
    return JudgeResponse(
        trace_id=trace_id,
        raw_text=raw,
        model_id=model,
        tokens_in=response.prompt_eval_count or 0,
        tokens_out=response.eval_count or 0,
        latency_s=latency
    )

def _call_uva(model: str, prompt: str, temperature: float, trace_id: str, base_url: str, system_prompt: str = "") -> JudgeResponse:
    """Call a model via the UvA AI Chat LiteLLM proxy (OpenAI-compatible)."""

    api_key = os.environ["UVA_API_KEY"]
    client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/") + "/v1")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    latency = time.perf_counter() - t0

    raw = response.choices[0].message.content or ""
    usage = response.usage
    return JudgeResponse(
        trace_id=trace_id,
        raw_text=raw,
        model_id=model,
        tokens_in=usage.prompt_tokens if usage else 0,
        tokens_out=usage.completion_tokens if usage else 0,
        latency_s=latency,
    )


def parse_14_modes(response: str):
    """
    Parse the LLM responses to extract yes/no answers for each failure mode.
    
    Args:
        responses: List of LLM responses evaluating traces
        
    Returns:
        Dictionary mapping failure mode codes to lists of binary values (0 for no, 1 for yes)
    """
    
    # Initialize dictionary with empty lists for each failure mode
    result = {}
    
    try:
            # Clean up the response - remove @@ markers if present
            cleaned_response = response.strip()
            if cleaned_response.startswith('@@'):
                cleaned_response = cleaned_response[2:]
            if cleaned_response.endswith('@@'):
                cleaned_response = cleaned_response[:-2]
            
            # Process each failure mode
            for mode in FAILURE_MODES:
                # Various patterns to match different response formats
                patterns = [
                    # Format with C. prefix and colon
                    rf"C\..*?{mode}.*?(yes|no)",
                    # Format with just C prefix without dot
                    rf"C{mode}\s+(yes|no)",
                    # Format with mode directly (with or without spaces)
                    rf"{mode}\s*[:]\s*(yes|no)",
                    rf"{mode}\s+(yes|no)",
                    # Format with newlines
                    rf"{mode}\s*\n\s*(yes|no)",
                    # Format with C prefix and newlines
                    rf"C\.{mode}\s*\n\s*(yes|no)"
                ]
                
                found = False
                for pattern in patterns:
                    matches = re.findall(pattern, cleaned_response, re.IGNORECASE | re.DOTALL)
                    if matches:
                        # Use the first match
                        value = 1 if matches[0].lower() == 'yes' else 0
                        result[mode] = value
                        found = True
                        break
                
                if not found:
                    # If we still can't find a match, try a more general approach
                    # Look for the mode number followed by any text and then yes/no
                    general_pattern = rf"(?:C\.)?{mode}.*?(yes|no)"
                    match = re.search(general_pattern, cleaned_response, re.IGNORECASE | re.DOTALL)
                    
                    if match:
                        value = 1 if match.group(1).lower() == 'yes' else 0
                        result[mode] = value
                    else:
                        # If all attempts fail, default to 'no'
                        print(f"Warning: Could not find mode {mode} in response")
                        result[mode] = 0
                    
    except Exception as e:
        print(f"Error processing response {e}")
        # If there's an error, default to 'no' for all modes for this response
        for mode in FAILURE_MODES:
            if mode not in result:  # Only append if we haven't already
                result[mode] = 0
    
    return result


def _stratified_sample(data: list[dict], n: int, key: str, seed: int = 42) -> list[dict]:
    """Sample n items proportionally per mas_name, with a fixed seed."""

    rng = random.Random(seed)
    by_mas_name: dict[str, list] = defaultdict(list)
    for item in data:
        by_mas_name[item[key]].append(item)

    total = len(data)
    result: list[dict] = []
    remainder: list[tuple[float, str]] = []

    for mas_name, items in by_mas_name.items():
        exact = n * len(items) / total
        quota = int(exact)
        result.extend(rng.sample(items, min(quota, len(items))))
        remainder.append((exact - quota, mas_name))

    leftover = n - len(result)
    sampled_ids = {id(x) for x in result}
    for _, mas_name in sorted(remainder, reverse=True)[:leftover]:
        pool = [x for x in by_mas_name[mas_name] if id(x) not in sampled_ids]
        if pool:
            chosen = rng.choice(pool)
            result.append(chosen)
            sampled_ids.add(id(chosen))

    rng.shuffle(result)
    return result


def load_dataset(config: JudgeConfig) -> list[dict]:
    with open(config.dataset_path) as f:
        data = json.load(f)
    if config.slice_n is not None and config.slice_n < len(data):
        data = _stratified_sample(data, config.slice_n, key="mas_name", seed=42)
    return data

_MODULE_DIR = Path(__file__).parent

class LLMJudge:
    def __init__(self, config: JudgeConfig):
        self.config = config
        self.definitions = Path(config.definitions_path).read_text()
        self.examples = Path(config.examples_path).read_text() if config.examples_path else ""

    def judge_trace(self, trace_id: str, trace_text: str) -> JudgeResponse:
        examples = self.examples if self.config.shots > 0 else ""
        prompt = build_judge_prompt(trace_text, self.definitions, examples)
        response = self._dispatch(prompt, trace_id)
        response.annotations = parse_14_modes(response.raw_text)
        return response

    def _dispatch(self, prompt, trace_id) -> JudgeResponse:
        if self.config.backend == "ollama":
            return _call_ollama(self.config.model, prompt, self.config.temperature, trace_id, self.config.ollama_host, self.config.system_prompt)
        if self.config.backend == "anthropic":
            return _call_anthropic_vertex(self.config.model, prompt, self.config.temperature, trace_id,
                    self.config.genai_project, self.config.genai_location, self.config.system_prompt, self.config.reasoning)
        if self.config.backend == "genai":
            return _call_genai(self.config.model, prompt, self.config.temperature, trace_id,
                    self.config.genai_project, self.config.genai_location, self.config.system_prompt, self.config.reasoning)
        if self.config.backend == "uva":
            return _call_uva(self.config.model, prompt, self.config.temperature, trace_id,
                    self.config.uva_base_url, self.config.system_prompt)
        raise ValueError(f"Unknown backend: {self.config.backend!r}. Use 'genai', 'anthropic', 'ollama', or 'uva'.")