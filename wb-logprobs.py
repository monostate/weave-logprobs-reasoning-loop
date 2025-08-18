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
# Install dependencies - handles both local and cloud environments
import subprocess
import sys
from pathlib import Path

# Check if we're in a local environment with vendorized polyfile-weave
local_polyfile = Path("./polyfile-weave")
if local_polyfile.exists() and local_polyfile.is_dir():
    print("Found local polyfile-weave, installing from vendorized source...")
    # Install local polyfile-weave first (with fixes for Python 3.9+ compatibility)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", "./polyfile-weave"])
    print("✓ Installed local polyfile-weave")

# Install remaining dependencies
try:
    import weave
    import openai
    import packaging
    print("✓ Required packages already installed")
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "weave", "openai", "packaging", "gql>=4.0.0", "set-env-colab-kaggle-dotenv"])
    print("✓ Installed required packages")

# %%
# Set your OpenAI API key
import os
from pathlib import Path

# Try to load from .env file if it exists
env_file = Path(".env")
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

# For notebooks, try set_env if available
try:
    from set_env import set_env
    _ = set_env("OPENAI_API_KEY")
except ImportError:
    pass

# %%
# Ensure OPENAI_API_KEY is set
if not os.environ.get("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not set. Please set it in your environment or .env file")
    print("Example: export OPENAI_API_KEY='sk-...'")

# %%
PROJECT = os.environ.get("WEAVE_PROJECT", "weave-intro-notebook")

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
# Apply runtime patch for gql 4.x compatibility by patching gql.Client directly
def patch_gql_client_for_v4():
    """
    Monkey-patch gql.Client.execute to handle the signature change between v3 and v4.
    This is a more direct approach that doesn't require modifying Weave's internal structure.
    """
    try:
        import gql
        from gql import Client
        from packaging import version
        
        # Check gql version
        GQL_VERSION = version.parse(gql.__version__ if hasattr(gql, '__version__') else '3.0.0')
        GQL_V4_PLUS = GQL_VERSION >= version.parse('4.0.0')
        
        print(f"Detected gql version: {GQL_VERSION}")
        
        if not GQL_V4_PLUS:
            print("gql 3.x detected, no patch needed")
            return True
        
        # Store original execute methods
        from gql.client import SyncClientSession, AsyncClientSession
        
        _orig_sync_execute = SyncClientSession.execute
        _orig_async_execute = AsyncClientSession.execute
        
        # Create wrapper that handles both call signatures
        def patched_sync_execute(self, document, *args, **kwargs):
            """Wrapper that accepts both v3 and v4 call signatures"""
            # If called with positional args (v3 style), convert to v4 style
            if args and 'variable_values' not in kwargs:
                # v3 style: execute(query, variables)
                kwargs['variable_values'] = args[0]
                return _orig_sync_execute(self, document, **kwargs)
            else:
                # v4 style or no variables
                return _orig_sync_execute(self, document, *args, **kwargs)
        
        async def patched_async_execute(self, document, *args, **kwargs):
            """Async wrapper that accepts both v3 and v4 call signatures"""
            # If called with positional args (v3 style), convert to v4 style
            if args and 'variable_values' not in kwargs:
                # v3 style: execute(query, variables)
                kwargs['variable_values'] = args[0]
                return await _orig_async_execute(self, document, **kwargs)
            else:
                # v4 style or no variables
                return await _orig_async_execute(self, document, *args, **kwargs)
        
        # Apply patches
        SyncClientSession.execute = patched_sync_execute
        AsyncClientSession.execute = patched_async_execute
        
        print("✓ Patched gql.Client for v3/v4 compatibility")
        return True
        
    except ImportError as e:
        print(f"ERROR: Could not import gql modules: {e}")
        raise RuntimeError(f"Failed to apply gql patch: {e}")
    except Exception as e:
        print(f"ERROR: Failed to patch gql.Client: {e}")
        raise RuntimeError(f"Failed to apply gql patch: {e}")

# Apply the patch BEFORE initializing Weave
patch_gql_client_for_v4()

# %%
# Initialize Weave - REQUIRED for tracking
weave.init(PROJECT)
print(f"✓ Weave initialized with project: {PROJECT}")

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
    tokens = []  # Store actual token text

    outputs = data.get("output") or data.get("outputs") or []
    if isinstance(outputs, list):
        for item in outputs:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for content in item.get("content", []) or []:
                if not isinstance(content, dict):
                    continue
                
                # Extract text
                if "text" in content:
                    text = content.get("text", text)
                
                # NEW: Handle the actual API format - logprobs is a LIST, not a dict!
                logprobs_list = content.get("logprobs")
                if isinstance(logprobs_list, list):
                    # Each item has: token, logprob, top_logprobs
                    for token_data in logprobs_list:
                        if isinstance(token_data, dict):
                            # Extract this token's logprob
                            token_text = token_data.get("token", "")
                            token_lp = token_data.get("logprob")
                            if token_lp is not None:
                                token_logprobs.append(float(token_lp))
                                tokens.append(token_text)
                            
                            # Extract top-k alternatives for this position
                            top_alts = token_data.get("top_logprobs", [])
                            alts = []
                            if isinstance(top_alts, list):
                                for alt in top_alts:
                                    if isinstance(alt, dict):
                                        alt_token = alt.get("token", "")
                                        alt_lp = alt.get("logprob")
                                        if alt_lp is not None:
                                            alts.append((alt_token, float(alt_lp)))
                            topk_by_pos.append(alts)

    # Return tokens as well for better analysis
    return text, token_logprobs, topk_by_pos, tokens

# %%
def _perplexity(token_logprobs):
    if not token_logprobs:
        # No logprobs means we can't calculate perplexity - don't trigger refinement
        return 0.0  # Low perplexity = high confidence
    if len(token_logprobs) == 0:
        return 0.0
    avg_logprob = sum(token_logprobs) / len(token_logprobs)
    # Perplexity = exp(-avg_logprob)
    # Lower perplexity = more confident
    # Higher perplexity = more uncertain
    return math.exp(-avg_logprob)

# %%
def _calculate_entropy(top_k_alternatives):
    """Calculate entropy from top-k alternatives at a position.
    Higher entropy = more uncertainty about which token to choose.
    """
    if not top_k_alternatives:
        return 0.0
    
    # Convert logprobs to probabilities
    probs = []
    for token, logprob in top_k_alternatives:
        probs.append(math.exp(logprob))
    
    # Normalize (they should sum to ~1 already for top token)
    total = sum(probs)
    if total == 0:
        return 0.0
    probs = [p/total for p in probs]
    
    # Calculate entropy: -sum(p * log(p))
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log(p)
    return entropy

def _uncertainty_report(token_logprobs, topk_by_pos, tokens=None, max_positions=10):
    if not token_logprobs:
        return "No token-level logprob info available."
    
    # Calculate entropy for each position
    entropies = []
    for i, alts in enumerate(topk_by_pos):
        if i < len(token_logprobs):
            entropies.append(_calculate_entropy(alts))
    
    # Calculate statistics
    avg_entropy = sum(entropies) / len(entropies) if entropies else 0
    max_entropy = max(entropies) if entropies else 0
    low_confidence_count = sum(1 for lp in token_logprobs if math.exp(lp) < 0.5)
    very_low_confidence_count = sum(1 for lp in token_logprobs if math.exp(lp) < 0.2)
    
    # Find most uncertain positions
    indices = list(range(len(token_logprobs)))
    # Sort by LOWEST logprob (most uncertain)
    indices.sort(key=lambda i: token_logprobs[i])
    
    lines = []
    lines.append("=== UNCERTAINTY ANALYSIS ===")
    lines.append(f"Total tokens: {len(token_logprobs)}")
    lines.append(f"Average entropy: {avg_entropy:.2f}")
    lines.append(f"Maximum entropy: {max_entropy:.2f}")
    lines.append(f"Low confidence tokens (<50%): {low_confidence_count}")
    lines.append(f"Very low confidence tokens (<20%): {very_low_confidence_count}")
    lines.append("")
    
    # Group uncertain tokens with context
    lines.append(f"Most uncertain tokens (top {min(max_positions, len(indices))}):")
    for rank, idx in enumerate(indices[:max_positions], 1):
        token_text = tokens[idx] if tokens and idx < len(tokens) else f"[pos {idx}]"
        lp = token_logprobs[idx]
        prob = math.exp(lp) * 100  # Convert to percentage
        entropy = entropies[idx] if idx < len(entropies) else 0.0
        
        # Get surrounding context (2 tokens before and after)
        context_before = ""
        context_after = ""
        if tokens:
            start = max(0, idx - 2)
            end = min(len(tokens), idx + 3)
            context_tokens = tokens[start:end]
            context_before = "".join(tokens[start:idx])
            context_after = "".join(tokens[idx+1:end])
        
        lines.append(f"\n  {rank}. Token: '{token_text}' (position {idx})")
        lines.append(f"     Context: ...{context_before}[{token_text}]{context_after}...")
        lines.append(f"     Confidence: {prob:.1f}%, Entropy: {entropy:.2f}")
        
        alts = topk_by_pos[idx] if idx < len(topk_by_pos) else []
        if alts:
            top_alts = []
            for tok, alt_lp in alts[:5]:  # Show top 5 alternatives
                alt_prob = math.exp(alt_lp) * 100
                top_alts.append(f"'{tok}' ({alt_prob:.1f}%)")
            lines.append(f"     Alternatives: {', '.join(top_alts)}")
    
    lines.append("\n=== KEY INSIGHTS ===")
    if very_low_confidence_count > 0:
        lines.append("- Multiple tokens with very low confidence detected")
    if max_entropy > 2.0:
        lines.append("- High entropy indicates multiple equally viable options")
    if low_confidence_count > len(token_logprobs) * 0.2:
        lines.append("- Over 20% of tokens have low confidence")
    
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
def first_pass_generation(question: str, model: str, temperature: float, top_k: int, is_reasoning_model: bool):
    """Generate initial response and extract logprobs/uncertainty metrics"""
    t_start = time.perf_counter()
    print(f"  [FIRST PASS] Starting generation with {model}...")
    
    create_params = {
        "model": model,
        "instructions": "You are a precise cryptography expert. Be concise and accurate.",
        "input": question,
    }
    
    # Only add temperature and logprobs for non-reasoning models
    if not is_reasoning_model:
        create_params["temperature"] = temperature
        create_params["top_logprobs"] = top_k
        create_params["include"] = ["message.output_text.logprobs"]
    
    print(f"  [FIRST PASS] Calling OpenAI API...")
    api_start = time.perf_counter()
    resp = client.responses.create(**create_params)
    api_end = time.perf_counter()
    print(f"  [FIRST PASS] API call took {api_end - api_start:.2f}s")
    
    print(f"  [FIRST PASS] Extracting metrics...")
    text, token_lps, topk_alts, tokens = _extract_text_and_logprobs(resp)
    in_tok, out_tok = _extract_usage(resp)
    ppx = _perplexity(token_lps)
    avg_lp = (sum(token_lps) / len(token_lps)) if token_lps else None
    table = _uncertainty_table(token_lps, topk_alts, max_positions=5)
    
    t_end = time.perf_counter()
    print(f"  [FIRST PASS] Total time: {t_end - t_start:.2f}s (tokens: {len(token_lps)}, perplexity: {ppx:.2f})")
    
    # Log ALL metrics to Weave
    return {
        "response": resp,
        "text": text,
        "tokens": tokens,  # NEW: actual token text
        "token_logprobs": token_lps,
        "top_k_alternatives": topk_alts,
        "perplexity": ppx,
        "avg_logprob": avg_lp,
        "uncertainty_table": table,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }

@weave.op()
def refinement_pass(question: str, draft_answer: str, model: str, temperature: float, 
                    top_k: int, ppx: float, token_lps: list, topk_alts: list, tokens: list):
    """Refine answer based on uncertainty analysis"""
    t_start = time.perf_counter()
    print(f"  [REFINEMENT] Starting refinement (perplexity was {ppx:.2f})...")
    
    analysis = _uncertainty_report(token_lps, topk_alts, tokens=tokens, max_positions=10)
    refined_input = (
        "You previously drafted an answer to a difficult question. "
        "Analysis shows you were uncertain about specific parts of your response.\n\n"
        f"Original Question: {question}\n\n"
        f"Your Draft Answer:\n{draft_answer}\n\n"
        f"Detailed Uncertainty Analysis (perplexity={ppx:.3f}):\n{analysis}\n\n"
        "REFINEMENT INSTRUCTIONS:\n"
        "1. Review the uncertain tokens and their alternatives\n"
        "2. Consider if any alternatives would be more accurate\n"
        "3. Pay special attention to tokens with <50% confidence\n"
        "4. For high-entropy tokens, choose the most factually accurate option\n"
        "5. Maintain the same structure but improve uncertain parts\n\n"
        "Provide a refined answer that resolves these uncertainties. "
        "Do not mention this analysis process in your response."
    )
    
    print(f"  [REFINEMENT] Input length: {len(refined_input)} chars")
    
    refine_params = {
        "model": model,
        "instructions": "You are a precise cryptography expert. Be concise and accurate.",
        "input": refined_input,
        "temperature": max(0.0, temperature - 0.1),
        "top_logprobs": top_k,
        "include": ["message.output_text.logprobs"]
    }
    
    print(f"  [REFINEMENT] Calling OpenAI API...")
    api_start = time.perf_counter()
    resp = client.responses.create(**refine_params)
    api_end = time.perf_counter()
    print(f"  [REFINEMENT] API call took {api_end - api_start:.2f}s")
    
    print(f"  [REFINEMENT] Extracting metrics...")
    text, token_lps, topk_alts, tokens = _extract_text_and_logprobs(resp)
    in_tok, out_tok = _extract_usage(resp)
    ppx = _perplexity(token_lps)
    avg_lp = (sum(token_lps) / len(token_lps)) if token_lps else None
    table = _uncertainty_table(token_lps, topk_alts, max_positions=5)
    
    t_end = time.perf_counter()
    print(f"  [REFINEMENT] Total time: {t_end - t_start:.2f}s (new perplexity: {ppx:.2f})")
    
    return {
        "response": resp,
        "text": text,
        "tokens": tokens,  # NEW: actual token text
        "token_logprobs": token_lps,
        "perplexity": ppx,
        "avg_logprob": avg_lp,
        "uncertainty_table": table,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "uncertainty_analysis": analysis,
    }

@weave.op()
def answer_difficult_question_with_uncertainty(
    question: str,
    model: str = "gpt-4.1-mini",
    top_k: int = 5,
    threshold: float = 1.4,
    temperature: float = 0.2,
):
    t0 = time.perf_counter()
    print(f"\n[MAIN] Starting uncertainty-aware generation for: '{question[:50]}...'")
    print(f"[MAIN] Model: {model}, Threshold: {threshold}")
    
    # Reasoning models (o1, o4) don't support temperature or logprobs
    is_reasoning_model = model.startswith(('o1', 'o4'))
    
    # First pass generation
    print(f"[MAIN] Starting first pass...")
    first_pass = first_pass_generation(question, model, temperature, top_k, is_reasoning_model)
    print(f"[MAIN] First pass complete.")
    resp1 = first_pass["response"]

    text1 = first_pass["text"]
    tokens1 = first_pass.get("tokens", [])  # Extract tokens
    token_lps1 = first_pass["token_logprobs"]
    topk1 = first_pass["top_k_alternatives"]
    in_tok1 = first_pass["input_tokens"]
    out_tok1 = first_pass["output_tokens"]
    ppx1 = first_pass["perplexity"]
    avg_lp1 = first_pass["avg_logprob"]
    table1 = first_pass["uncertainty_table"]

    did_refine = False
    final_text = text1
    refinement_data = None
    ppx2 = None
    avg_lp2 = None
    table2 = None
    in_tok2, out_tok2 = None, None

    # Calculate additional uncertainty metrics for better decision
    max_entropy = 0.0
    high_uncertainty_tokens = 0
    if token_lps1 and topk1:
        for i, (lp, alts) in enumerate(zip(token_lps1, topk1)):
            entropy = _calculate_entropy(alts)
            max_entropy = max(max_entropy, entropy)
            # Count tokens with <50% confidence
            if math.exp(lp) < 0.5:
                high_uncertainty_tokens += 1
    
    # Skip refinement for reasoning models (no logprobs available)
    print(f"[MAIN] Checking if refinement needed...")
    print(f"  - Perplexity: {ppx1:.2f} (threshold: {threshold})")
    print(f"  - Max entropy: {max_entropy:.2f}")
    print(f"  - High uncertainty tokens: {high_uncertainty_tokens}")
    
    # Refine if ANY uncertainty metric is high
    should_refine = (
        (ppx1 > threshold) or 
        (max_entropy > 1.5) or  # High entropy = multiple viable options
        (high_uncertainty_tokens >= 3)  # Multiple uncertain tokens
    )
    
    if should_refine and not is_reasoning_model:
        print(f"[MAIN] Refinement triggered! Starting refinement pass...")
        did_refine = True
        refinement_data = refinement_pass(
            question, text1, model, temperature, top_k, ppx1, token_lps1, topk1, tokens1
        )
        
        text2 = refinement_data["text"]
        ppx2 = refinement_data["perplexity"]
        avg_lp2 = refinement_data["avg_logprob"]
        final_text = text2
        table2 = refinement_data["uncertainty_table"]
        in_tok2 = refinement_data["input_tokens"]
        out_tok2 = refinement_data["output_tokens"]
        print(f"[MAIN] Refinement complete.")
    else:
        print(f"[MAIN] No refinement needed (ppx={ppx1:.2f} <= {threshold} or reasoning model)")
    
    t1 = time.perf_counter()
    print(f"[MAIN] TOTAL TIME: {t1 - t0:.2f}s")

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
                "max_entropy": max_entropy,
                "high_uncertainty_tokens": high_uncertainty_tokens,
            },
            "refinement": {
                "enabled": did_refine,
                "triggered_by": "perplexity" if ppx1 > threshold else ("entropy" if max_entropy > 1.5 else "uncertain_tokens"),
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
            "max_entropy": max_entropy,
            "high_uncertainty_tokens": high_uncertainty_tokens,
            "uncertainty_table": table1,
            "input_tokens": in_tok1,
            "output_tokens": out_tok1,
        },
        "refinement": {
            "enabled": did_refine,
            "triggered_by": "perplexity" if did_refine and ppx1 > threshold else ("entropy" if did_refine and max_entropy > 1.5 else ("uncertain_tokens" if did_refine else None)),
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

# %%
# Test with a controversial question (runs in both notebook and script)
print("Running uncertainty-aware generation test...\n")

# Test question
test_question = "Is artificial general intelligence likely to be achieved by 2030?"

print(f"{'='*60}")
print(f"Question: {test_question}")
print('='*60)

# Run with non-reasoning model (uncertainty loop)
print("\n▶ GPT-4.1-mini with uncertainty loop:")
base_result = answer_difficult_question_with_uncertainty(
    test_question,
    model="gpt-4.1-mini",
    top_k=5,
    threshold=1.4,
    temperature=0.2,
)

# Truncate answer for display
answer_preview = base_result.get('final_answer', '')[:300] + "..." if len(base_result.get('final_answer', '')) > 300 else base_result.get('final_answer', '')
print(f"  Answer: {answer_preview}")
print(f"  Perplexity: {base_result['first_pass'].get('perplexity', 'N/A'):.3f}")
print(f"  Max Entropy: {base_result['first_pass'].get('max_entropy', 'N/A'):.3f}")
print(f"  High Uncertainty Tokens: {base_result['first_pass'].get('high_uncertainty_tokens', 'N/A')}")
print(f"  Refinement: {'✓ Triggered' if base_result['refinement']['enabled'] else '✗ Not needed'}")
if base_result['refinement']['enabled']:
    print(f"  Triggered by: {base_result['refinement'].get('triggered_by', 'N/A')}")
print(f"  Cost: ${base_result['usage'].get('estimated_cost_usd', 0):.4f}")

# Run with reasoning model for comparison
print("\n▶ o4-mini (reasoning model):")
reasoning_result = answer_difficult_question_with_uncertainty(
    test_question,
    model="o4-mini",
    top_k=5,
    threshold=1.4,
    temperature=0.2,
)

# Truncate answer for display
reason_answer_preview = reasoning_result.get('final_answer', '')[:300] + "..." if len(reasoning_result.get('final_answer', '')) > 300 else reasoning_result.get('final_answer', '')
print(f"  Answer: {reason_answer_preview}")
print(f"  Cost: ${reasoning_result['usage'].get('estimated_cost_usd', 0):.4f}")

# Compare efficiency
if reasoning_result['usage'].get('estimated_cost_usd', 0) > 0:
    cost_ratio = base_result['usage'].get('estimated_cost_usd', 0) / reasoning_result['usage'].get('estimated_cost_usd', 1)
    print(f"\n  💰 Cost efficiency: {cost_ratio:.1%} of reasoning model cost")

print("\n" + "="*60)
print("✅ Check Weave UI for detailed analysis of logprobs and uncertainty metrics!")
print("="*60)