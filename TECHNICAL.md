# Technical Documentation: Uncertainty-Aware Generation

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [Uncertainty Metrics](#uncertainty-metrics)
3. [Decision Logic](#decision-logic)
4. [Implementation Details](#implementation-details)
5. [API Response Processing](#api-response-processing)
6. [Observability with Weave](#observability-with-weave)

## Core Concepts

### The Uncertainty Loop Architecture

```
Input Question → First Pass Generation → Extract Logprobs → Calculate Metrics
                                                               ↓
                    Final Answer ← Refinement Pass ← Trigger Decision
```

## Uncertainty Metrics

### 1. Perplexity

**Definition**: Perplexity measures how "surprised" the model is by its own predictions.

**Formula**:
```python
def _perplexity(token_logprobs):
    if not token_logprobs:
        return 0.0  # No logprobs = high confidence (prevents unnecessary refinement)
    avg_logprob = sum(token_logprobs) / len(token_logprobs)
    return math.exp(-avg_logprob)
```

**Interpretation**:
- **Low perplexity (≈1.0)**: Model is confident
- **High perplexity (>2.0)**: Model is uncertain
- **Threshold**: 1.4 (empirically determined)

### 2. Token-Level Entropy

**Definition**: Entropy quantifies uncertainty in the probability distribution over alternative tokens.

**Formula**:
```python
def _calculate_entropy(top_k_alternatives):
    """
    Calculate Shannon entropy: H = -Σ(p_i * log(p_i))
    Higher entropy = more uncertainty about which token to choose
    """
    if not top_k_alternatives:
        return 0.0
    
    # Convert logprobs to probabilities
    probs = [math.exp(logprob) for token, logprob in top_k_alternatives]
    
    # Normalize probabilities
    total = sum(probs)
    if total == 0:
        return 0.0
    probs = [p/total for p in probs]
    
    # Calculate entropy
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log(p)
    return entropy
```

**Interpretation**:
- **Entropy = 0**: Complete certainty (100% confidence in one token)
- **Entropy ≈ 0.7**: Moderate uncertainty (≈50/50 between two options)
- **Entropy > 1.5**: High uncertainty (multiple viable options)
- **Maximum entropy**: log(k) where k is number of alternatives

### 3. Confidence Distribution

**Definition**: Statistical analysis of token confidence levels across the response.

**Metrics Tracked**:
```python
# Count tokens below confidence thresholds
low_confidence_count = sum(1 for lp in token_logprobs if math.exp(lp) < 0.5)      # <50%
very_low_confidence_count = sum(1 for lp in token_logprobs if math.exp(lp) < 0.2) # <20%

# Percentage of uncertain tokens
uncertainty_ratio = low_confidence_count / len(token_logprobs)
```

## Decision Logic

### Multi-Metric Refinement Trigger

The system triggers refinement if ANY of these conditions are met:

```python
should_refine = (
    (perplexity > 1.4) or                    # Overall uncertainty high
    (max_entropy > 1.5) or                   # At least one very uncertain token
    (high_uncertainty_tokens >= 3)           # Multiple uncertain tokens
)
```

### Why Multiple Metrics?

Each metric captures different uncertainty patterns:

| Metric | Detects | Example Scenario |
|--------|---------|------------------|
| **Perplexity** | Overall uncertainty | Model unsure throughout response |
| **Max Entropy** | Single critical uncertainty | One key fact/number uncertain |
| **Token Count** | Distributed uncertainty | Multiple small uncertainties |

### Real Examples from Testing

```
Question: "Is artificial general intelligence likely to be achieved by 2030?"

Metrics:
- Perplexity: 1.35 (below threshold)
- Max entropy: 1.43 (below threshold)  
- High uncertainty tokens: 8 (ABOVE threshold)
Result: Refinement triggered due to multiple uncertain tokens
```

## Implementation Details

### Extracting Logprobs from OpenAI Responses API

The API returns logprobs in a specific format that must be parsed correctly:

```python
def _extract_text_and_logprobs(resp):
    """Extract from actual API format (logprobs is a LIST, not dict)"""
    token_logprobs = []
    topk_by_pos = []
    tokens = []
    
    # Navigate response structure
    outputs = data.get("output") or []
    for item in outputs:
        if item.get("type") == "message":
            for content in item.get("content", []):
                logprobs_list = content.get("logprobs")
                if isinstance(logprobs_list, list):
                    # Each item has: token, logprob, top_logprobs
                    for token_data in logprobs_list:
                        token_text = token_data.get("token", "")
                        token_lp = token_data.get("logprob")
                        if token_lp is not None:
                            token_logprobs.append(float(token_lp))
                            tokens.append(token_text)
                        
                        # Extract alternatives
                        top_alts = token_data.get("top_logprobs", [])
                        alts = []
                        for alt in top_alts:
                            alt_token = alt.get("token", "")
                            alt_lp = alt.get("logprob")
                            if alt_lp is not None:
                                alts.append((alt_token, float(alt_lp)))
                        topk_by_pos.append(alts)
    
    return text, token_logprobs, topk_by_pos, tokens
```

### Uncertainty Report Generation

The system generates detailed reports showing specific uncertain tokens with context:

```python
def _uncertainty_report(token_logprobs, topk_by_pos, tokens, max_positions=10):
    """Generate human-readable uncertainty analysis"""
    
    # Statistical summary
    lines = [
        "=== UNCERTAINTY ANALYSIS ===",
        f"Total tokens: {len(token_logprobs)}",
        f"Average entropy: {avg_entropy:.2f}",
        f"Maximum entropy: {max_entropy:.2f}",
        f"Low confidence tokens (<50%): {low_confidence_count}",
        f"Very low confidence tokens (<20%): {very_low_confidence_count}"
    ]
    
    # Token-specific analysis with context
    for rank, idx in enumerate(most_uncertain_positions):
        token = tokens[idx]
        confidence = math.exp(token_logprobs[idx]) * 100
        
        # Show surrounding context
        context_before = "".join(tokens[idx-2:idx])
        context_after = "".join(tokens[idx+1:idx+3])
        
        lines.append(f"""
  {rank}. Token: '{token}' (position {idx})
     Context: ...{context_before}[{token}]{context_after}...
     Confidence: {confidence:.1f}%, Entropy: {entropy:.2f}
     Alternatives: {top_alternatives}
        """)
    
    return "\n".join(lines)
```

### Example Uncertainty Report

```
=== UNCERTAINTY ANALYSIS ===
Total tokens: 62
Average entropy: 0.45
Maximum entropy: 1.43
Low confidence tokens (<50%): 8
Very low confidence tokens (<20%): 2

Most uncertain tokens (top 10):

  1. Token: 'likely' (position 15)
     Context: ...is [likely] to...
     Confidence: 32.1%, Entropy: 1.43
     Alternatives: 'unlikely' (28.5%), 'possible' (22.3%), 'expected' (10.1%)

  2. Token: '2030' (position 28)
     Context: ...by [2030] is...
     Confidence: 41.2%, Entropy: 1.21
     Alternatives: '2040' (31.5%), '2035' (15.8%), '2050' (8.3%)
```

## API Response Processing

### Performance Characteristics

Our testing revealed important API behavior:

| Question Type | Tokens | API Response Time | Model |
|--------------|--------|-------------------|-------|
| Simple (capital) | 8 | 1.3s | gpt-4.1-mini |
| Complex (P vs NP) | 465 | 67s | gpt-4.1-mini |
| Complex (P vs NP) | 568 | 99s | gpt-4.1 |
| Complex (P vs NP) | 414 | 61s | gpt-4o |

**Key Finding**: Response time scales with complexity, not just token count. The API appears to spend more time on complex technical content.

### Adaptive Analysis Size

For long responses, we limit the uncertainty analysis to prevent huge refinement prompts:

```python
# Scale analysis based on response length
max_analysis_positions = min(10, max(5, 50 // len(token_lps))) if token_lps else 10
```

- Short responses (<10 tokens): Analyze all
- Medium responses (10-50 tokens): Analyze up to 10
- Long responses (>50 tokens): Analyze 5-10 most uncertain

## Observability with Weave

### Hierarchical Operation Tracking

The implementation uses nested `@weave.op()` decorators for comprehensive tracking:

```python
@weave.op()
def answer_difficult_question_with_uncertainty(...):
    # Main orchestrator - logs overall metrics
    
    @weave.op()
    def first_pass_generation(...):
        # Logs: tokens, logprobs, perplexity, entropy
        return {
            "tokens": tokens,
            "token_logprobs": token_lps,
            "perplexity": ppx,
            "max_entropy": max_entropy,
            "uncertainty_table": table
        }
    
    @weave.op()
    def refinement_pass(...):
        # Logs: uncertainty analysis, refinement results
        return {
            "uncertainty_analysis": analysis,
            "perplexity_after": ppx2,
            # ... refinement metrics
        }
```

### Weave Summary Attachment

Critical metrics are attached for easy UI inspection:

```python
current_call = weave.require_current_call()
current_call.summary = {
    "first_pass": {
        "perplexity": ppx1,
        "max_entropy": max_entropy,
        "high_uncertainty_tokens": high_uncertainty_tokens
    },
    "refinement": {
        "enabled": did_refine,
        "triggered_by": "perplexity" if ppx1 > threshold 
                       else ("entropy" if max_entropy > 1.5 
                       else "uncertain_tokens")
    }
}
```

### Queryable Metrics in Weave UI

After execution, the Weave UI enables:

1. **Filtering runs by uncertainty level**
   - `first_pass.perplexity > 1.5`
   - `refinement.enabled == True`

2. **Comparing refinement impact**
   - Before/after perplexity
   - Cost vs quality tradeoffs

3. **Analyzing uncertainty patterns**
   - Which tokens are consistently uncertain
   - Correlation between entropy and errors

4. **Building uncertainty datasets**
   - Export high-uncertainty examples
   - Create training data for uncertainty-aware models

## Cost-Benefit Analysis

### Refinement Economics

| Scenario | First Pass | Refinement | Total Cost | Quality |
|----------|------------|------------|------------|---------|
| Simple question | $0.0003 | None | $0.0003 | ✓ Good |
| Uncertain response | $0.0007 | $0.0004 | $0.0011 | ✓✓ Improved |
| o4-mini (reasoning) | N/A | N/A | $0.0025 | ✓✓ Good |

**Result**: 2.75x cost reduction vs reasoning models with comparable quality

## Future Enhancements

### Planned Improvements

1. **Dynamic Thresholds**
   - Adjust based on question complexity
   - Learn optimal thresholds per domain

2. **Token Importance Weighting**
   - Weight uncertainty by semantic importance
   - Focus on factual vs stylistic tokens

3. **Streaming Refinement**
   - Detect uncertainty during generation
   - Interrupt and refine mid-stream

4. **Uncertainty Calibration**
   - Fine-tune models to output calibrated uncertainties
   - Train on collected uncertainty patterns

## References

- Shannon, C. E. (1948). "A Mathematical Theory of Communication"
- Guo et al. (2017). "On Calibration of Modern Neural Networks"
- Malinin & Gales (2018). "Predictive Uncertainty Estimation via Prior Networks"
