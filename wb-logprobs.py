# -*- coding: utf-8 -*-
# %% [markdown]
# # Weave + OpenAI Responses API: uncertainty-aware generation
# 
# This notebook demonstrates how to:
# - Call the OpenAI Responses API with `include=["message.output_text.logprobs"]` and `top_logprobs`.
# - Compute token-level uncertainty (perplexity) from logprobs.
# - If uncertainty is high, run a refinement pass that informs the model about uncertain regions and top-k alternatives.
# - Log all inputs, outputs, and metrics to Weave using `@weave.op` so you can inspect traces and compare iterations.
# 
# Prereqs: set `OPENAI_API_KEY` in your environment and install `weave` and `openai`.

# %%
# %%capture
# !pip install weave openai set-env-colab-kaggle-dotenv

# %%

# Set your OpenAI API key
from set_env import set_env

# %%
# Put your OPENAI_API_KEY in the secrets panel to the left 🗝️
_ = set_env("OPENAI_API_KEY")
# os.environ["OPENAI_API_KEY"] = "sk-..." # alternatively, put your key here

# %%
PROJECT = "weave-intro-notebook"

# %% [markdown]
# ## What we'll build
# 
# - A single `@weave.op` function that implements the uncertainty-aware loop with the Responses API.
# - The op returns structured metrics (average logprob, perplexity, whether refinement ran) and the final answer.
# - You can iterate on thresholds, `top_logprobs`, or prompts and compare runs in Weave.

# %%

from openai import OpenAI
import json
import math
import time

# %%
import weave
import os

# %%
# Initialize Weave with error handling for compatibility issues
try:
    weave.init(PROJECT)
    print(f"Weave initialized with project: {PROJECT}")
except TypeError as e:
    # Fallback for compatibility issues with gql library
    print(f"Note: Weave initialization encountered an issue: {e}")
    print("Attempting alternative initialization...")
    try:
        # Try with explicit entity/project format
        import wandb
        entity = os.environ.get("WANDB_ENTITY", wandb.api.default_entity())
        if entity:
            weave.init(f"{entity}/{PROJECT}")
        else:
            # Run without W&B integration
            print("Running without full Weave tracking. Results will be local only.")
            os.environ["WEAVE_DISABLED"] = "true"
    except Exception as fallback_error:
        print(f"Weave tracking disabled due to: {fallback_error}")
        print("The notebook will run but without experiment tracking.")
        os.environ["WEAVE_DISABLED"] = "true"

# %%
client = OpenAI()

# %%
def _extract_text_and_logprobs(resp):
    """Extract output text, per-token logprobs, and top-k alternatives from a Responses API result."""
    # Try to get a plain dict
    try:
        data = resp.model_dump()
    except Exception:
        try:
            data = json.loads(resp.json())
        except Exception:
            data = {}

    text = getattr(resp, "output_text", None) or ""
    token_logprobs = []
    topk_by_pos = []

    outputs = data.get("output") or data.get("outputs") or []
    if isinstance(outputs, list):
        for item in outputs:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for content in item.get("content", []) or []:
                if not isinstance(content, dict):
                    continue
                logprobs_payload = None
                if "output_text" in content and isinstance(content["output_text"], dict):
                    inner = content["output_text"]
                    text = inner.get("text", text)
                    logprobs_payload = inner.get("logprobs")
                else:
                    if "text" in content:
                        text = content.get("text", text)
                    logprobs_payload = content.get("logprobs")

                if isinstance(logprobs_payload, dict):
                    raw_lps = logprobs_payload.get("token_logprobs") or []
                    if isinstance(raw_lps, list):
                        token_logprobs = [float(x) for x in raw_lps if x is not None]
                    raw_top = logprobs_payload.get("top_logprobs")
                    if isinstance(raw_top, list):
                        parsed = []
                        for pos in raw_top:
                            alts = []
                            if isinstance(pos, list):
                                for alt in pos:
                                    if isinstance(alt, dict):
                                        tok = alt.get("token") or alt.get("text") or ""
                                        lp = alt.get("logprob")
                                        if lp is not None:
                                            alts.append((str(tok), float(lp)))
                            parsed.append(alts)
                        topk_by_pos = parsed

    return text, token_logprobs, topk_by_pos

# %%
def _perplexity(token_logprobs):
    if not token_logprobs:
        return float("inf")
    return math.exp(-sum(token_logprobs) / max(1, len(token_logprobs)))

# %%
def _uncertainty_report(token_logprobs, topk_by_pos, max_positions=5):
    if not token_logprobs:
        return "No token-level logprob info available."
    indices = list(range(len(token_logprobs)))
    indices.sort(key=lambda i: token_logprobs[i])
    lines = []
    for idx in indices[:max_positions]:
        alts = topk_by_pos[idx] if idx < len(topk_by_pos) else []
        formatted = ", ".join(f"{tok} ({lp:.2f})" for tok, lp in alts)
        lines.append(f"pos {idx}: lp={token_logprobs[idx]:.2f}; alts: {formatted}")
    return "\n".join(lines)

# %%
def _uncertainty_table(token_logprobs, topk_by_pos, max_positions=5):
    """Return a compact table of the most uncertain positions and their top-k alternatives.

    Shape: [{"position": int, "token_logprob": float, "top_k": [{"token": str, "logprob": float}]}]
    """
    if not token_logprobs:
        return []
    indices = list(range(len(token_logprobs)))
    indices.sort(key=lambda i: token_logprobs[i])
    rows = []
    for idx in indices[:max_positions]:
        alts = []
        if idx < len(topk_by_pos):
            for alt in (topk_by_pos[idx] or []):
                if isinstance(alt, tuple):
                    tok, lp = alt
                elif isinstance(alt, dict):
                    tok, lp = alt.get("token"), alt.get("logprob")
                else:
                    tok, lp = None, None
                if tok is not None and lp is not None:
                    alts.append({"token": str(tok), "logprob": float(lp)})
        rows.append(
            {
                "position": idx,
                "token_logprob": float(token_logprobs[idx]),
                "top_k": alts,
            }
        )
    return rows

# %%
# Simple cost table (USD) per 1M tokens
PRICING_USD_PER_MILLION = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "o4-mini": {"input": 1.10, "output": 4.40},
}

# %%
def _extract_usage(resp):
    """Return (input_tokens, output_tokens) if present, else (None, None)."""
    try:
        data = resp.model_dump()
    except Exception:
        try:
            data = json.loads(resp.json())
        except Exception:
            data = {}

    usage = data.get("usage") or {}
    # The Responses API commonly provides input_tokens/output_tokens
    in_tok = usage.get("input_tokens")
    out_tok = usage.get("output_tokens")
    if isinstance(in_tok, int) and isinstance(out_tok, int):
        return in_tok, out_tok
    # Fallbacks if the shape differs
    in_tok = usage.get("prompt_tokens") or usage.get("input_tokens_total")
    out_tok = usage.get("completion_tokens") or usage.get("output_tokens_total")
    if isinstance(in_tok, int) and isinstance(out_tok, int):
        return in_tok, out_tok
    return None, None

# %%
def _estimate_cost_usd(model: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    pricing = PRICING_USD_PER_MILLION.get(model)
    if not pricing or input_tokens is None or output_tokens is None:
        return None
    return (input_tokens / 1_000_000.0) * pricing["input"] + (output_tokens / 1_000_000.0) * pricing["output"]

# %%
def _extract_reasoning_metadata(resp):
    """Extract basic reasoning metadata if present: count and encrypted length.

    Returns dict(reasoning_items: int | None, encrypted_chars: int | None)
    """
    try:
        data = resp.model_dump()
    except Exception:
        try:
            data = json.loads(resp.json())
        except Exception:
            data = {}

    outputs = data.get("output") or data.get("outputs") or []
    if not isinstance(outputs, list):
        return {"reasoning_items": None, "encrypted_chars": None}

    reasoning_items = 0
    encrypted_chars = 0
    for item in outputs:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "reasoning":
            continue
        reasoning_items += 1
        for content in item.get("content", []) or []:
            if isinstance(content, dict):
                enc = content.get("encrypted_content")
                if isinstance(enc, str):
                    encrypted_chars += len(enc)

    return {"reasoning_items": reasoning_items, "encrypted_chars": encrypted_chars}

# %% [markdown]
# ## Weave-logged uncertainty loop
# 
# In this section we wrap the uncertainty-aware generation into a single `@weave.op`. Weave will:
# - Log function code, inputs (question and parameters), and outputs.
# - Capture nested OpenAI calls (first pass and optional refinement).
# - Let you inspect token logprobs, perplexity, and compare experiments.

# %%
@weave.op()
def answer_difficult_question_with_uncertainty(
    question: str,
    model: str = "gpt-4.1-mini",
    top_k: int = 5,
    threshold: float = 1.4,
    temperature: float = 0.2,
):
    t0 = time.perf_counter()
    
    # Reasoning models (o1, o4) don't support temperature parameter
    is_reasoning_model = model.startswith(('o1', 'o4'))
    
    create_params = {
        "model": model,
        "instructions": "You are a precise cryptography expert. Be concise and accurate.",
        "input": question,
        "top_logprobs": top_k,
        "include": ["message.output_text.logprobs"],
    }
    
    # Only add temperature for non-reasoning models
    if not is_reasoning_model:
        create_params["temperature"] = temperature
    
    resp1 = client.responses.create(**create_params)

    text1, token_lps1, topk1 = _extract_text_and_logprobs(resp1)
    in_tok1, out_tok1 = _extract_usage(resp1)
    ppx1 = _perplexity(token_lps1)
    avg_lp1 = (sum(token_lps1) / len(token_lps1)) if token_lps1 else None
    table1 = _uncertainty_table(token_lps1, topk1, max_positions=5)

    did_refine = False
    final_text = text1
    ppx2 = None
    avg_lp2 = None

    if ppx1 > threshold:
        did_refine = True
        analysis = _uncertainty_report(token_lps1, topk1, max_positions=5)
        refined_input = (
            "You previously drafted an answer to a difficult question.\n"
            f"Question: {question}\n\n"
            f"Your draft answer:\n{text1}\n\n"
            f"Uncertainty analysis (perplexity={ppx1:.3f}):\n{analysis}\n\n"
            "Revise the answer to improve factual accuracy and clarity, resolving uncertain parts. "
            "Do not mention this analysis in the final answer."
        )

        refine_params = {
            "model": model,
            "instructions": "You are a precise cryptography expert. Be concise and accurate.",
            "input": refined_input,
            "top_logprobs": top_k,
            "include": ["message.output_text.logprobs"],
        }
        
        # Only add temperature for non-reasoning models
        if not is_reasoning_model:
            refine_params["temperature"] = max(0.0, temperature - 0.1)
        
        resp2 = client.responses.create(**refine_params)

        text2, token_lps2, _ = _extract_text_and_logprobs(resp2)
        in_tok2, out_tok2 = _extract_usage(resp2)
        ppx2 = _perplexity(token_lps2)
        avg_lp2 = (sum(token_lps2) / len(token_lps2)) if token_lps2 else None
        final_text = text2
        table2 = _uncertainty_table(token_lps2, topk1, max_positions=5)
    else:
        table2 = None
        in_tok2, out_tok2 = None, None
    t1 = time.perf_counter()

    # Attach a concise summary to the Weave call for easy inspection in the UI
    try:
        current_call = weave.require_current_call()
        current_call.summary = {
            "parameters": {
                "model": model,
                "top_k": top_k,
                "threshold": threshold,
                "temperature": temperature,
            },
            "first_pass": {
                "token_count": len(token_lps1),
                "avg_logprob": avg_lp1,
                "perplexity": ppx1,
            },
            "refinement": {
                "enabled": did_refine,
                "avg_logprob_after": avg_lp2,
                "perplexity_after": ppx2,
            },
            "timing_seconds": t1 - t0,
        }
    except Exception:
        # If not in a tracked context, just skip summary attachment
        pass

    # Aggregate token usage/costs
    total_input_tokens = (in_tok1 or 0) + (in_tok2 or 0)
    total_output_tokens = (out_tok1 or 0) + (out_tok2 or 0)
    estimated_cost_usd = _estimate_cost_usd(model, total_input_tokens, total_output_tokens)

    return {
        "question": question,
        "final_answer": final_text,
        "first_pass": {
            "answer": text1,
            "avg_logprob": avg_lp1,
            "perplexity": ppx1,
            "uncertainty_table": table1,
            "input_tokens": in_tok1,
            "output_tokens": out_tok1,
        },
        "refinement": {
            "enabled": did_refine,
            "perplexity_after": ppx2,
            "avg_logprob_after": avg_lp2,
            "uncertainty_table_after": table2,
            "input_tokens_after": in_tok2,
            "output_tokens_after": out_tok2,
        },
        "parameters": {
            "model": model,
            "top_k": top_k,
            "threshold": threshold,
            "temperature": temperature,
        },
        "usage": {
            "total_input_tokens": total_input_tokens if (in_tok1 is not None or in_tok2 is not None) else None,
            "total_output_tokens": total_output_tokens if (out_tok1 is not None or out_tok2 is not None) else None,
            "estimated_cost_usd": estimated_cost_usd,
            "timing_seconds": t1 - t0,
        },
        "model_kind": "reasoning" if is_reasoning_model else "non_reasoning",
    }

# %%
# Ask a difficult question and log everything to Weave via the op above
question = (
    "What are the implications of P vs NP for modern cryptography? Provide concrete examples and caveats."
)
with weave.attributes({
    "tag": "uncertainty-loop",
    "question": question,
    "question_topic": "cryptography",
    "variant": "gpt-4.1-mini",
}):
    base_result = answer_difficult_question_with_uncertainty(
        question,
        model="gpt-4.1-mini",
        top_k=5,
        threshold=1.4,
        temperature=0.2,
    )

# %%
with weave.attributes({
    "tag": "uncertainty-loop",
    "question": question,
    "question_topic": "cryptography",
    "variant": "o4-mini",
}):
    reasoning_result = answer_difficult_question_with_uncertainty(
        question,
        model="o4-mini",
        top_k=5,
        threshold=1.4,
        temperature=0.2,
    )

# %%
def _fmt_cost(x):
    return f"${x:.4f}" if isinstance(x, (int, float)) and x is not None else "-"

# %%
print("\n==== Final Answers ====")
print("[gpt-4.1-mini]\n", base_result.get("final_answer", ""))
print("\n[o4-mini]\n", reasoning_result.get("final_answer", ""))

# %%
print("\n==== Usage/Cost/Time (estimates) ====")
base_usage = base_result.get("usage", {})
reason_usage = reasoning_result.get("usage", {})
print(
    "gpt-4.1-mini:",
    "tokens(in/out)=",
    (base_usage.get("total_input_tokens"), base_usage.get("total_output_tokens")),
    "cost=",
    _fmt_cost(base_usage.get("estimated_cost_usd")),
    "time(s)=",
    base_usage.get("timing_seconds"),
)
print(
    "o4-mini:",
    "tokens(in/out)=",
    (reason_usage.get("total_input_tokens"), reason_usage.get("total_output_tokens")),
    "cost=",
    _fmt_cost(reason_usage.get("estimated_cost_usd")),
    "time(s)=",
    reason_usage.get("timing_seconds"),
)

# %% [markdown]
# ## End
# 
# This notebook now focuses on the Weave-logged uncertainty-aware generation loop using the OpenAI Responses API.
# Use the Weave UI links to explore traces, inputs/outputs, and compare iterations.